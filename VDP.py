import torch
import torch.nn as nn
import neuromancer.psl as psl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
from callback import CallbackVDP
from TrainingViz.viz import Viz


seed = 100
nepochs = 30
lambda_l1 = 0.
# ground truth system model
gt_model = psl.nonautonomous.VanDerPolControl()
# sampling rate
ts = gt_model.params[1]['ts']
# problem dimensions
nx = gt_model.nx    # number of states
nu = gt_model.nu    # number of control inputs
nref = nx           # number of references
# constraints bounds
umin = -5.
umax = 5.
xmin = -4.
xmax = 4.

# white-box ODE model with no-plant model mismatch
van_der_pol = ode.VanDerPolControl()
van_der_pol.mu = nn.Parameter(torch.tensor(gt_model.mu), requires_grad=False)

# integrate continuous time ODE
integrator = integrators.RK4(van_der_pol, h=torch.tensor(ts))
integrator_node = Node(integrator, ['xn', 'u'], ['xn'], name='model')

nsteps = 15  # prediction horizon
n_samples = 50    # number of sampled scenarios
nsteps_test = 100

x_test_size = (20, nsteps_test+1, nx)
u_test_size = (20, nsteps_test, nu)
torch.manual_seed(seed)
test_d = {'xn': torch.randn(20, 1, nx, dtype=torch.float32),
          'r': torch.zeros(20, nsteps_test+1, nx, dtype=torch.float32),
          'xmin': torch.full(x_test_size, xmin),
          'xmax': torch.full(x_test_size, xmax),
          'umin': torch.full(u_test_size, umin),
          'umax': torch.full(u_test_size, umax)}


def get_policy_data(nsteps, n_samples, nsteps_test):
    dev_samples = 100
    dev_bs = 1

    x_train_size = (n_samples, nsteps+1, nx)
    x_dev_size = (dev_samples, nsteps+1, nx)
    x_test_size = (1, nsteps_test+1, nx)

    u_train_size = (n_samples, nsteps, nu)
    u_dev_size = (dev_samples, nsteps, nu)
    u_test_size = (1, nsteps_test, nu)

    # Training dataset generation
    train_d = DictDataset({'xn': torch.randn(n_samples, 1, nx),
                           'r': torch.zeros(n_samples, nsteps+1, nx),
                           'xmin': torch.full(x_train_size, xmin),
                           'xmax': torch.full(x_train_size, xmax),
                           'umin': torch.full(u_train_size, umin),
                           'umax': torch.full(u_train_size, umax)},
                          name='train')
    # Development dataset generation
    dev_d = DictDataset({'xn': torch.randn(dev_samples, 1, nx),
                         'r': torch.zeros(dev_samples, nsteps+1, nx),
                         'xmin': torch.full(x_dev_size, xmin),
                         'xmax': torch.full(x_dev_size, xmax),
                         'umin': torch.full(u_dev_size, umin),
                         'umax': torch.full(u_dev_size, umax)},
                        name='dev')

    test_d = DictDataset({'xn': torch.randn(1, 1, nx, dtype=torch.float32),
                          'r': torch.zeros(1, nsteps_test+1,
                                           nx, dtype=torch.float32),
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
objectives = [regulation_loss, loss_l1]
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


callb = CallbackVDP(relocation=False)

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
overfit = callb.overfit_data

y_key = 'test_xn'
keys = {'r_key': 'test_r',
        'u_key': 'test_u',
        'ymin_key': 'test_xmin',
        'ymax_key': 'test_xmax',
        'umin_key': 'test_umin',
        'umax_key': 'test_umax',
        'loss_key': 'test_loss'}

cl_system.nsteps = nsteps_test

outputs = cl_system(test_d)

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
cl_system_r = System([policy_r, integrator_node], nsteps=nsteps)
components = [cl_system_r]
# create constrained optimization loss
# construct constrained optimization problem
problem = Problem(components, loss)

optimizer = torch.optim.AdamW(policy_r.parameters(), lr=0.002)

callb = CallbackVDP(relocation=True)
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
overfit_r = callb.overfit_data

cl_system_r.nsteps = nsteps_test

outputs_r = cl_system_r(test_d)


mean_train_loss_no = torch.stack(overfit["mean_train_loss"]).detach().numpy()
mean_dev_loss_no = torch.stack(overfit["mean_dev_loss"]).numpy()

mean_train_loss_r = torch.stack(overfit_r["mean_train_loss"]).detach().numpy()
mean_dev_loss_r = torch.stack(overfit_r["mean_dev_loss"]).numpy()

plt.plot(mean_train_loss_no, color="blue", label="Train Loss (No Relocation)")
plt.plot(mean_dev_loss_no, color="blue", label="Dev Loss (No Relocation)",
         linestyle="--")

plt.plot(mean_train_loss_r, color="red", label="Train Loss (Relocation)")
plt.plot(mean_dev_loss_r, color="red", linestyle="--",
         label="Dev_Loss (Relocation)")
plt.xlabel("Epoch")
plt.ylabel("Mean Loss")
plt.legend()
plt.savefig("Attempt1.png", dpi=300)
plt.show()

tracking_mse = torch.nn.functional.mse_loss(outputs['xn'], outputs['r'])
tracking_mse_r = torch.nn.functional.mse_loss(outputs_r['xn'], outputs_r['r'])

print(f"MSE: {tracking_mse.item()}")
print(f"MSE (R): {tracking_mse_r.item()}")


dist = torch.squeeze(torch.stack(callb.dist))

start = dist[0]
fig, ax = plt.subplots(2, figsize=(25, 25))
title = fig.suptitle("Distribution of Initial Conditions", fontsize=48)


hist_top = ax[0].plot(start[:, 0], color="black")
hist_bot = ax[1].plot(start[:, 1], color="red")


def update(frame):
    frame_y = dist[frame, :, :]
    plt.cla()
    ax[0].plot(frame_y[:, 0], color="black")
    ax[1].plot(frame_y[:, 1], color="red")


ani = animation.FuncAnimation(fig=fig, func=update, frames=nepochs,
                              interval=500)

ani.save(filename="test.gif", writer="pillow")
