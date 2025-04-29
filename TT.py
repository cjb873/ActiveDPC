import torch
import numpy as np
import torch.nn as nn
import neuromancer.psl as psl
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.modules.activations import activations
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.dynamics import ode, integrators
from neuromancer.loggers import BasicLogger
from TrainingViz.trainer import VizTrainer
from callback import CallbackTT
from TrainingViz.viz import Viz

seed = 22
nepochs = 30
lambda_l1 = 0.25
# ground truth system model
gt_model = psl.nonautonomous.TwoTank(seed=seed)
# sampling rate
ts = gt_model.params[1]['ts']
# problem dimensions
nx = gt_model.nx    # number of states
nu = gt_model.nu    # number of control inputs
nref = nx           # number of references
# constraints bounds
umin = 0.
umax = 1.
xmin = 0.
xmax = 1.
sys = gt_model


# white-box ODE model with no-plant model mismatch
two_tank = ode.TwoTankParam()
two_tank.c1 = nn.Parameter(torch.tensor(gt_model.c1), requires_grad=False)
two_tank.c2 = nn.Parameter(torch.tensor(gt_model.c2), requires_grad=False)
# integrate continuous time ODE
integrator = integrators.RK4(two_tank, h=torch.tensor(ts))
# symbolic system model
integrator_node = Node(integrator, ['xn', 'u'], ['xn'], name='model')
bounds_node = Node(lambda x: torch.clamp(x, xmin, xmax), ['x_un'], ['xn'],
                   name='bounds')


def get_policy_data(nsteps, n_samples, nsteps_test):
    dev_samples = 50
    dev_bs = 1

    x_train_size = (n_samples, nsteps+1, nx)
    x_dev_size = (dev_samples, nsteps+1, nx)
    x_test_size = (1, nsteps_test+1, nx)

    u_train_size = (n_samples, nsteps, nu)
    u_dev_size = (dev_samples, nsteps, nu)
    u_test_size = (1, nsteps_test, nu)

    list_refs = [torch.rand(1, 1)*torch.ones(nsteps+1, nref)
                 for k in range(n_samples)]
    ref = torch.cat(list_refs)
    batched_ref = ref.reshape([n_samples, nsteps+1, nref])

    # Training dataset generation
    train_d = DictDataset({'xn': torch.randn(n_samples, 1, nx),
                           'r': batched_ref,
                           'xmin': torch.full(x_train_size, xmin),
                           'xmax': torch.full(x_train_size, xmax),
                           'umin': torch.full(u_train_size, umin),
                           'umax': torch.full(u_train_size, umax)},
                          name='train')
    list_refs = [torch.rand(1, 1)*torch.ones(nsteps+1, nref)
                 for k in range(dev_samples)]
    ref = torch.cat(list_refs)
    batched_ref = ref.reshape([dev_samples, nsteps+1, nref])
    # Development dataset generation
    dev_d = DictDataset({'xn': torch.randn(dev_samples, 1, nx),
                         'r': batched_ref,
                         'xmin': torch.full(x_dev_size, xmin),
                         'xmax': torch.full(x_dev_size, xmax),
                         'umin': torch.full(u_dev_size, umin),
                         'umax': torch.full(u_dev_size, umax)},
                        name='dev')

    # generate reference
    np_refs = psl.signals.step(nsteps_test+1, 1, min=xmin, max=xmax,
                               randsteps=5, rng=np.random.default_rng(seed))
    R = torch.tensor(np_refs, dtype=torch.float32).reshape(1, nsteps_test+1, 1)
    torch_ref = torch.cat([R, R], dim=-1)

    test_d = DictDataset({'xn': torch.randn(1, 1, nx, dtype=torch.float32),
                          'r': torch_ref,
                          'xmin': torch.full(x_test_size, xmin),
                          'xmax': torch.full(x_test_size, xmax),
                          'umin': torch.full(u_test_size, umin),
                          'umax': torch.full(u_test_size, umax)},
                         name='test')

    # torch dataloaders
    batch_size = 200
    train_loader = torch.utils.data.DataLoader(train_d,
                                               batch_size=batch_size,
                                               collate_fn=train_d.collate_fn,
                                               shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_d,
                                             batch_size=dev_bs,
                                             collate_fn=dev_d.collate_fn,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_d,
                                              batch_size=1,
                                              collate_fn=test_d.collate_fn,
                                              shuffle=False)

    return train_loader, dev_loader, test_loader


torch.manual_seed(seed)
nsteps = 50  # prediction horizon
n_samples = 2000    # number of sampled scenarios
nsteps_test = 750
train_loader, dev_loader, test_loader = \
        get_policy_data(nsteps, n_samples, nsteps_test)

# neural net control policy with hard control action bounds
net = blocks.MLP_bounds(insize=nx + nref, outsize=nu, hsizes=[32, 32],
                        nonlin=activations['gelu'], min=umin, max=umax)
policy = Node(net, ['xn', 'r'], ['u'], name='policy')

# closed-loop system model
cl_system = System([policy, integrator_node], nsteps=nsteps)


x = variable('xn')
ref = variable('r')
x_min = variable('xmin')
x_max = variable('xmax')
l1 = variable([x], lambda x: torch.norm(list(policy.parameters())[0], 1))

loss_l1 = lambda_l1*(l1 == 0)

# objectives
regulation_loss = 5. * ((x == ref) ^ 2)  # target posistion
# constraints
state_lower_bound_penalty = 10.*(x > x_min)
state_upper_bound_penalty = 10.*(x < x_max)

# objectives and constraints names for nicer plot
regulation_loss.name = 'state_loss'
state_lower_bound_penalty.name = 'x_min'
state_upper_bound_penalty.name = 'x_max'

# list of constraints and objectives
objectives = [regulation_loss]
constraints = [
    state_lower_bound_penalty,
    state_upper_bound_penalty,
]

components = [cl_system]
# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = Problem(components, loss)

optimizer = torch.optim.AdamW(policy.parameters(), lr=0.002)


callb = CallbackTT(relocation=False)

logger = BasicLogger(args=None, verbosity=1,
                     stdout=['dev_loss', 'train_loss'])

print("Without Relocation")
#  Neuromancer trainer
trainer = VizTrainer(
    problem,
    train_loader,
    dev_loader,
    test_loader,
    optimizer=optimizer,
    epochs=nepochs,
    train_metric='train_loss',
    eval_metric='dev_loss',
    test_metric='test_loss',
    warmup=5,
    callback=callb,
    logger=logger
)
# Train control policy
best_model = trainer.train()
# load best trained model
trainer.model.load_state_dict(best_model)

data = callb.get_data()

y_key = 'test_xn'
keys = {'r_key': 'test_r',
        'u_key': 'test_u',
        'ymin_key': 'test_xmin',
        'ymax_key': 'test_xmax',
        'umin_key': 'test_umin',
        'umax_key': 'test_umax',
        'loss_key': 'test_loss'}

v = Viz(data, y_key, **keys)

v.animate('without_relocation_tt')
overfit = callb.overfit_data
print(overfit)
print("\n\n\nWith Relocation:")
torch.manual_seed(seed)
train_loader, dev_loader, test_loader = \
        get_policy_data(nsteps, n_samples, nsteps_test)
net_r = blocks.MLP_bounds(insize=nx + nref, outsize=nu, hsizes=[32, 32],
                          nonlin=activations['gelu'], min=umin, max=umax)

policy_r = Node(net_r, ['xn', 'r'], ['u'], name='policy')

x = variable('xn')
ref = variable('r')
x_min = variable('xmin')
x_max = variable('xmax')
l1 = variable([x], lambda x: torch.norm(list(policy_r.parameters())[0], 1))

loss_l1 = lambda_l1*(l1 == 0)

# objectives
regulation_loss = 5. * ((x == ref) ^ 2)  # target posistion
# constraints
state_lower_bound_penalty = 10.*(x > x_min)
state_upper_bound_penalty = 10.*(x < x_max)

# objectives and constraints names for nicer plot
regulation_loss.name = 'state_loss'
state_lower_bound_penalty.name = 'x_min'
state_upper_bound_penalty.name = 'x_max'

# list of constraints and objectives
objectives = [regulation_loss, loss_l1]
constraints = [
    state_lower_bound_penalty,
    state_upper_bound_penalty,
]

# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)
# closed-loop system model
cl_system = System([policy_r, integrator_node], nsteps=nsteps)
components = [cl_system]
# construct constrained optimization problem
problem = Problem(components, loss)

optimizer = torch.optim.AdamW(policy_r.parameters(), lr=0.002)

callb = CallbackTT(relocation=True)
#  Neuromancer trainer
trainer = VizTrainer(
    problem,
    train_loader,
    dev_loader,
    test_loader,
    optimizer=optimizer,
    epochs=nepochs,
    train_metric='train_loss',
    eval_metric='dev_loss',
    test_metric='test_loss',
    warmup=5,
    callback=callb,
    logger=logger
)
# Train control policy
best_model = trainer.train()
trainer.model.load_state_dict(best_model)

data = callb.get_data()
overfit = callb.overfit_data
print(overfit)

v = Viz(data, y_key, **keys)

v.animate('with_relocation_tt')
