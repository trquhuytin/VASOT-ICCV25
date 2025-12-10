import numpy as np
import torch
import torch.nn.functional as F


# Fast implmentation of Cv @ T @ Ck.T for GW component of ASOT with O(NK) complexity


# This func creates a conv1d filter called weights, where the center frame is set to 0, and its adjacent frames(on both sides) are set to 1/r
# Frames within a radius of N*r frames are considered adjacent
# Finally convert the 1-D tensor into a 3-D tensor and return it
# e.g. N=256, r=0.02, abs_r=5, weights=3-D tensor where 3rd dim has 11 elements with 0 at index 5 and 1/r everywhere else
def construct_Ca_filter(K, r, device):
    abs_r = int(K * r)
    weights = torch.ones((2 * abs_r) + (2 * K) + 1, device=device)
    weights[K:K + abs_r] = 0                      # First block of abs_r zeros
    weights[K + abs_r + 1:K + 2 * abs_r + 1] = 0  # Second block of abs_r zeros
    return weights[None, None, :]

def construct_Cv_filter(N, r, device):
    abs_r = int(N * r)
    weights = torch.ones(2 * abs_r + 1, device=device) / r
    weights[abs_r] = 0.
    return weights[None, None, :]


def mult_Cv(Cv_weights, X):
    # X is equivalent to T*Ca
    B, N, K = X.shape
    # Use conv1d as a shortcut to compute Cv * (T*Ca), using the previously constructed filter Cv
    Y_flat = F.conv1d(X.transpose(1, 2).reshape(-1, 1, N), Cv_weights, padding='same')
    # Returns Cv*T*Ca with the shape (B,N,K)
    return Y_flat.reshape(B, K, N).transpose(1, 2)


# ASOT objective function gradients for mirro descent solver

# NOTE: This is used to compute the FGW objective from eq (3), as well as to compute the gradient or derivative of eq (3) w.r.t. T  
def grad_fgw(T, cost_matrix, alpha, Cv):

    # Shortcut for computing T*Ca, where T.shape=(B,N,K) and Ca.shape=(B,K,K) and Ca would have 0s in the diagonal and 1s everywhere else
    T_Ck = T.sum(dim=2, keepdim=True) - T
    # Returns alpha*(Cv*T*Ca) + (1-alpha)*(Ck)
    return alpha * mult_Cv(Cv, T_Ck) + (1. - alpha) * cost_matrix

def grad_kld(T, p, lambd, axis):
    # p is marginal, dim is marginal axes
    marg = T.sum(dim=axis, keepdim=True)
    return lambd * (torch.log(marg / p + 1e-12) + 1.)

def grad_entropy(T, eps):
    return - torch.log(T + 1e-12) * eps


# Sinkhorn projections for balanced ASOT (balanced assignment for frames AND actions)

def project_to_polytope_KL(cost_matrix, mask, eps, dx, dy, n_iters=10, stable_thres=7.):
    # runs sinkhorn algorithm on dual potentials w/log domain stabilization
    dev = cost_matrix.device
    B, N, K = cost_matrix.shape
    dual_pot = torch.exp(-cost_matrix / eps) * mask.unsqueeze(2)
    dual_pot = dual_pot / dual_pot.max()
    b = torch.ones((B, K, 1), device=dev)
    u = torch.zeros((B, N, 1), device=dev)
    v = torch.zeros((K, 1), device=dev)

    for i in range(n_iters):
        a = dx / (dual_pot @ b)
        a = torch.nan_to_num(a, posinf=0., neginf=0.)
        b = dy / (dual_pot.transpose(1, 2) @ a)
        b = torch.nan_to_num(b, posinf=0., neginf=0.)
        if torch.any(torch.log(a).abs() > stable_thres) or torch.any(torch.log(b).abs() > stable_thres):
            if i != n_iters - 1:
                u = torch.nan_to_num(u + eps * torch.log(a), posinf=0., neginf=0.)
                v = torch.nan_to_num(v + eps * torch.log(b), posinf=0., neginf=0.)
                dual_pot = torch.exp((u + v.transpose(1, 2) - cost_matrix) / eps) * mask.unsqueeze(2)
                b = torch.ones_like(b)
    T = a * dual_pot * b.transpose(1, 2)
    return T


# ASOT objective function evaluation

def kld(a, b, eps=1e-10):
    return (a * torch.log(a / b + eps)).sum(dim=1)


def entropy(T, eps=1e-10):
    return (-T * torch.log(T + eps) + T).sum(dim=(1, 2))


def ot_objective(T, cost_matrix, eps, alpha, radius, ub_frames, ub_actions,
                   lambda_frames, lambda_actions, mask=None):
    dev = cost_matrix.device
    B, N, K = cost_matrix.shape
        
    if mask is None:
        mask = torch.full((B, N), 1, dtype=bool, device=dev)
    nnz = mask.sum(dim=1)
    T_mask = T * mask.unsqueeze(2)
    # The code above is the same as in segment_asot except we don't need to initialize T since it already exists
    # We'll use the masked T for most of the obj, and only use T to calculate the entropy regularization term −ϵH(T) 

    ## FGW stuff    
    # NOTE: Read my comments in construct_Cv_filter() and grad_fgw() for more clarity
    # These steps are an efficient way to compute the FGW objective, eq (3) from ASOT
    # 1. Cv is a Conv1d filter(we don't explicitly compute Cv and Ca)
    # 2. Compute grad_fgw = alpha*(Cv*T*Ca) + (1-alpha)*(Ck)
    # 3. Element-wise multiply it with T
    # 4. Finally, sum all the elements in each batch element, to get a single value(fgw_obj) per batch element
    # (Steps 3 and 4 are a shortcut to compute the 2 dot/inner products with T, apply alpha weights, and add them)
    # fgw_obj = (alpha)*<Cv*T*Ca, T> + (1-alpha)*<Ck, T>
    Cv = construct_Cv_filter(N, radius, dev)
    
    fgw_obj = (grad_fgw(T_mask, cost_matrix, alpha, Cv) * T_mask).sum(dim=(1, 2))
    
    # Unbalanced stuff (if ub_frames or ub_actions are True)
    # The code from here till right before entr is for the KLD term in eq (4)
    dy = torch.ones((B, K), device=dev) / K
    dx = torch.ones((B, N), device=dev) / nnz[:, None]
    
    frames_marg = T_mask.sum(dim=2)
    frames_ub_penalty = kld(frames_marg, dx) * lambda_frames
    actions_marg = T_mask.sum(dim=1)
    actions_ub_penalty = kld(actions_marg, dy) * lambda_actions
    
    # The KLD term(ub) of eq(4) is initially 0, this would be the value used for balanced OT
    ub = torch.zeros(B, device=dev)
    # But if either of the 2 flags below are True then it becomes an unbalanced OT and ub is modified
    if ub_frames:
        ub += frames_ub_penalty
    if ub_actions:
        ub += actions_ub_penalty
    
    # entropy regularization term −ϵH(T) from section 4.4
    entr = -eps * entropy(T)
    
    # objective
    obj = 0.5 * fgw_obj + ub + entr
    
    return obj


# ASOT solver

def segment_ot(cost_matrix, mask=None,
                 eps=0.07, alpha=0.3, radius=0.04, ub_frames=False,
                 ub_actions=True, lambda_frames=0.1, lambda_actions=0.05,
                 n_iters=(25, 1), stable_thres=7., step_size=None):
                # saveflag=None):
    
    # Get the device of this tensor
    dev = cost_matrix.device
    # B is batch size, N is no. of frames, K is no. of actions
    B, N, K = cost_matrix.shape

    # If no mask, make a mask of shape BxN and fill it with Trues 
    if mask is None:
        mask = torch.full((B, N), 1, dtype=bool, device=dev)

    # Compute the no. of Trues in the mask for each video in the batch
    # (maybe to get the no. of unique frames in each video, instead of using N which includes padding) TODO??? 
    nnz = mask.sum(dim=1)

    # dy and dx are q and p from section 4.1
    # ones normalized by K, to represent the uniform distribution over actions 
    dy = torch.ones((B, K, 1), device=dev) / K
    # tensor of ones normalized by nnz, to represent the uniform distribution over frames, 
    dx = torch.ones((B, N, 1), device=dev) / nnz[:, None, None]

    # Initialize the Transportation Matrix according to Appendix A
    # Compute T for a vid, by setting every element to 1/N*K but here N=nnz. Repeat for every vid in the batch
    T = dx * dy.transpose(1, 2)
    # Apply the mask to T to deal with the duplicate frames(where mask is False)
    # for each frame i where mask[i] is False, set all elements in row i of T to 0
    T = T * mask.unsqueeze(2)
    
    # Read comments in the function definition
    Cv = construct_Cv_filter(N, radius, dev)

    
# ############DE-BUG BUG BUG BUG########################################
# ## Make Cv and Ca here and just save it for a viz, dont use anywhere
# ## I dont think it requires any thing that depends on training so 
# ## we will exit after 1 training step after we have 6 matrices
# ## Save a Cv and Ca for 2 seg and 1 align(copy code to alignment_asot.py)
# ## and make sure each of them has a unique name to prevent overwriting when saving
# ## Maybe pass a flag to the segment_asot() to indicate whether its for vid X or vid Y

#     MY_Nr = int(round(N * radius))

#     # Compute Cv
#     MY_Cv = np.zeros((N, N))
#     for i in range(N):
#         for k in range(N):
#             if 1 <= abs(i - k) <= MY_Nr:
#                 MY_Cv[i, k] = 1 / radius

#     # Compute Ca
#     MY_Ca = np.ones((K, K))  # Default to 1
#     np.fill_diagonal(MY_Ca, 0)  # Set diagonal to 0

#     # Save as .npy files
#     np.save(f"{saveflag}_Cv.npy", MY_Cv)
#     np.save(f"{saveflag}_Ca.npy", MY_Ca)
# ############DE-BUG BUG BUG BUG########################################
    
    # the trace stores all of the computed objs(transportation costs)
    obj_trace = []
    # Iteration number for the outer loop
    it = 0

    while True:
        with torch.no_grad():
            # Compute the transportation cost for the current transportation matrix T, using the ASOT objective function
            obj = ot_objective(T, cost_matrix, eps, alpha, radius, ub_frames, ub_actions,
                                lambda_frames, lambda_actions, mask=mask)
        # Append the current obj to the trace
        obj_trace.append(obj)
        
        # Exit the loop when all outer iterations are completed
        if it >= n_iters[0]:
            break
        
        # TODO: Might have to add/remove gradient terms when we make any changes above this
        # gradient of objective function required for mirror descent step
        fgw_cost_matrix = grad_fgw(T, cost_matrix, alpha, Cv)
        grad_obj = fgw_cost_matrix - grad_entropy(T, eps)
        if ub_frames:
            grad_obj += grad_kld(T, dx, lambda_frames, 2)
        if ub_actions:
            grad_obj += grad_kld(T, dy.transpose(1, 2), lambda_actions, 1)
        
        # automatically calibrate stepsize by rescaling based on observed gradient
        if it == 0 and step_size is None:
            step_size = 4. / grad_obj.max().item()

        # update step - note, no projection required if both sides are unbalanced
        # Update T like you'd update weights in gradient descent 
        T = T * torch.exp(-step_size * grad_obj)
        
        # 1st condition is used for the balanced OT, the other 2 conditions are for unbalanced OT
        if not ub_frames and not ub_actions:
            T = project_to_polytope_KL(fgw_cost_matrix, mask, eps, dx, dy,
                                       n_iters=n_iters[1], stable_thres=stable_thres)
        elif not ub_frames:
            T /= T.sum(dim=2, keepdim=True)
            T = torch.nan_to_num(T, posinf=0., neginf=0.)
            T = T * dx
        elif not ub_actions:
            T /= T.sum(dim=1, keepdim=True)
            T = torch.nan_to_num(T, posinf=0., neginf=0.)
            T = T * dy.transpose(1, 2)
        
        it += 1
    
    T = T * nnz[:, None, None]  # rescale so marginals per frame sum to 1
    obj_trace = torch.cat(obj_trace)
    return T, obj_trace

# ρR = rho * Temporal prior
def temporal_prior(n_frames, n_clusters, rho, device):
    temp_prior = torch.abs(torch.arange(n_frames)[:, None] / n_frames - torch.arange(n_clusters)[None, :] / n_clusters).to(device)
    return rho * temp_prior
