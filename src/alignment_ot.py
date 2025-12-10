import numpy as np
import torch
import torch.nn.functional as F

# Fast implementation of Cv @ T @ Ca for GW component of ASOT with O(NK) complexity


# This func creates a conv1d filter called weights, where the center frame is set to 0, and its adjacent frames(on both sides) are set to 1/r
# Frames within a radius of N*r frames are considered adjacent
# Finally convert the 1-D tensor into a 3-D tensor and return it
# e.g. N=256, r=0.02, abs_r=5, weights=3-D tensor where 3rd dim has 11 elements with 0 at index 5 and 1/r everywhere else
def construct_Cv_filter(N, r, device):
    abs_r = int(N * r)
    weights = torch.ones(2 * abs_r + 1, device=device) / r
    weights[abs_r] = 0.
    return weights[None, None, :]

# Use conv1d as a shortcut to compute Cv * (T*Ca), using the previously created Conv1D filter Cv
def mult_Cv(Cv_weights, X):
    # X is equivalent to T*Ca
    B, N, K = X.shape
    
    # Shape = B, K, N
    # rows become the cols and cols become the rows
    X = X.transpose(1, 2)
    
    # Shape = B×K,1,N
    # Takes the columns from each matrix across the batch elements, and combines them into a big batch of columns
    X = X.reshape(-1, 1, N)

    # The conv. filter will be applied to each column(of size N) separately, moving from top to bottom of the column while producing an output column of the same size
    # (basically applies the filter to each row of the original input X, with padding)
    Y_flat = F.conv1d(X, Cv_weights, padding='same')

    # It reverses the reshape and transpose operations to get the same shape as the original input X
    # Returns Cv*T*Ca with the shape (B,N,K)
    return Y_flat.reshape(B, K, N).transpose(1, 2)

def construct_Ca_filter(K, r, device):
    abs_r = int(K * r)
    weights = torch.ones((2 * abs_r) + (2 * K) + 1, device=device)
    weights[K:K + abs_r] = 0                      # First block of abs_r zeros
    weights[K + abs_r + 1:K + 2 * abs_r + 1] = 0  # Second block of abs_r zeros
    return weights[None, None, :]

# Use conv1d as a shortcut to compute T * Ca, using the previously created Conv1D filter Ca
def mult_Ca(Ca_weights, X):
    # X is equivalent to T
    B, N, K = X.shape

    # Shape = B×N,1,K
    # Takes the columns from each matrix across the batch elements, and combines them into a big batch of columns  
    X = X.reshape(-1, 1, K)

    # The conv. filter will be applied to each column(of size K) separately, moving from top to bottom of the column while producing an output column of the same size
    # (basically applies the filter to each column of the original input X, with padding)
    # NOTE: Moving X and Ca_weights to cpu or changing their dtype to float64 gives the exact same result as matrix multiplication
    Y_flat = F.conv1d(X, Ca_weights, padding=(Ca_weights.size(-1) - 1) // 2)

    # It reverses the reshape operation to get the same shape as the original input X
    # Returns T*Ca with the shape (B,N,K)
    return Y_flat.reshape(B, N, K)


# ASOT objective function gradients for mirro descent solver

# NOTE: This is used to compute the FGW objective from eq (3), as well as to compute the gradient or derivative of eq (3) w.r.t. T  
def grad_fgw(T, cost_matrix, alpha, Cv, Ca):
    
    # Calculate T*Ca
    T_Ca = mult_Ca(Ca, T)
    # Returns alpha*(Cv*T*Ca) + (1-alpha)*(Ck)
    return alpha * mult_Cv(Cv, T_Ca) + (1. - alpha) * cost_matrix

def grad_kld(T, p, lambd, axis):
    # p is marginal, dim is marginal axes
    marg = T.sum(dim=axis, keepdim=True)
    return lambd * (torch.log(marg / p + 1e-12) + 1.)

def grad_entropy(T, eps):
    return - torch.log(T + 1e-12) * eps


# Sinkhorn projections for balanced ASOT (balanced assignment for frames AND actions)

def project_to_polytope_KL(cost_matrix, mask_X, mask_Y, eps, dx, dy, n_iters=10, stable_thres=7.):
    # runs sinkhorn algorithm on dual potentials w/log domain stabilization
    dev = cost_matrix.device
    B, N, K = cost_matrix.shape
    dual_pot = torch.exp(-cost_matrix / eps) * mask_X.unsqueeze(2) * mask_Y.unsqueeze(1)
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
                dual_pot = torch.exp((u + v.transpose(1, 2) - cost_matrix) / eps) * mask_X.unsqueeze(2) * mask_Y.unsqueeze(1)
                b = torch.ones_like(b)
    T = a * dual_pot * b.transpose(1, 2)
    return T


# ASOT objective function evaluation

def kld(a, b, eps=1e-10):
    return (a * torch.log(a / b + eps)).sum(dim=1)


def entropy(T, eps=1e-10):
    return (-T * torch.log(T + eps) + T).sum(dim=(1, 2))


def ot_objective(T, cost_matrix, eps, alpha, radius, ub_frames, ub_actions,
                   lambda_frames, lambda_actions, mask_X=None, mask_Y=None):
    dev = cost_matrix.device
    B, N, K = cost_matrix.shape
        
    if mask_X is None:
        mask_X = torch.full((B, N), 1, dtype=bool, device=dev)
    if mask_Y is None:
        mask_Y = torch.full((B, N), 1, dtype=bool, device=dev)

    nnz_X = mask_X.sum(dim=1)
    nnz_Y = mask_Y.sum(dim=1)

    T_mask = T * mask_X.unsqueeze(2) * mask_Y.unsqueeze(1)
    # The code above is the same as in segment_asot except we don't need to initialize T since it already exists
    # We'll use the masked T for most of the obj, and only use T to calculate the entropy regularization term −ϵH(T) 

    ## FGW stuff    
    # NOTE: Read my comments in construct_Cv_filter() and grad_fgw() for more clarity
    # These steps are an efficient way to compute the FGW objective, eq (3) from ASOT
    # 1. Cv and Ca are Conv1d filters(we don't explicitly compute them as matrices)
    # 2. Compute grad_fgw = alpha*(Cv*T*Ca) + (1-alpha)*(Ck)
    # 3. Element-wise multiply it with T
    # 4. Finally, sum all the elements in each batch element, to get a single value(fgw_obj) per batch element
    # (Steps 3 and 4 are a shortcut to compute the 2 dot/inner products with T, apply alpha weights, and add them)
    # fgw_obj = (alpha)*<Cv*T*Ca, T> + (1-alpha)*<Ck, T>
    Cv = construct_Cv_filter(N, radius, dev)
    Ca = construct_Ca_filter(K, radius, dev)

    fgw_obj = (grad_fgw(T_mask, cost_matrix, alpha, Cv, Ca) * T_mask).sum(dim=(1, 2))
    
    # Unbalanced stuff (if ub_frames or ub_actions are True)
    # The code from here till right before entr is for the KLD term in eq (4)
    dy = torch.ones((B, K), device=dev) / nnz_Y[:, None]
    dx = torch.ones((B, N), device=dev) / nnz_X[:, None]
    
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

def align_ot(cost_matrix, mask_X=None, mask_Y=None,
                 eps=0.07, alpha=0.3, radius=0.04, ub_frames=False,
                 ub_actions=True, lambda_frames=0.1, lambda_actions=0.05,
                 n_iters=(25, 1), stable_thres=7., step_size=None):
                #  saveflag=None):

    # print(lambda_frames)
    # Get the device of this tensor
    dev = cost_matrix.device
    # B is batch size, N is no. of frames in video X, K is no. of frames in video Y
    B, N, K = cost_matrix.shape

    # For simplicity we keep the 2 masks separate throughout the code
    # Use mask_X to mask the rows and mask_Y to mask the cols of T
    # If a mask is None, then make a mask of shape BxN and fill it with Trues 
    if mask_X is None:
        mask_X = torch.full((B, N), 1, dtype=bool, device=dev)
    if mask_Y is None:
        mask_Y = torch.full((B, N), 1, dtype=bool, device=dev)

    mask_X = torch.cat((torch.ones(1, 1, dtype=torch.bool).to(dev), mask_X), dim=1)
    mask_X[:,0] = True

    mask_Y = torch.cat((torch.ones(1,1, dtype=torch.bool).to(dev), mask_Y), dim=1)
    mask_Y[:,0] = True


    # When computing p and q from section 4.1, we need to use the actual no. of frames in each video without padding
    # Count the no. of frames in each video X in this batch, excluding the padding frames(duplicates of the final frame where mask contains a False).
    nnz_X = mask_X.sum(dim=1)
    # Repeat the above for each video Y in this batch
    nnz_Y = mask_Y.sum(dim=1)

    # dy and dx are q and p respectively from section 4.1
    # tensor of ones normalized by nnz_Y, to represent the uniform distribution over frames from video Y  
    dy = torch.ones((B, K, 1), device=dev) / nnz_Y[:, None, None]
    # tensor of ones normalized by nnz_X, to represent the uniform distribution over frames from video X
    dx = torch.ones((B, N, 1), device=dev) / nnz_X[:, None, None]

    # Initialize the Transportation Matrix according to Appendix A
    # Compute the T for each batch element, by setting every element of T to 1/N*K, but here N=nnz_X and K=nnz_Y due to padding
    T = dx * dy.transpose(1, 2)

    # Apply the mask to T to deal with the padding frames, where mask is False
    # for each frame i of video X where mask_X[i] is False, set all elements in row i of T to 0
    # for each frame j of video Y where mask_Y[j] is False, set all elements in col j of T to 0
    T = T * mask_X.unsqueeze(2) * mask_Y.unsqueeze(1)
    
    # Read comments in the function definition
    Cv = construct_Cv_filter(N, radius, dev)
    Ca = construct_Ca_filter(K, radius, dev)

# ############DE-BUG BUG BUG BUG########################################
# ## Make Cv and Ca here and just save it for a viz, dont use anywhere
# ## I dont think it requires any thing that depends on training so 
# ## we will exit after 1 training step after we have 6 matrices
# ## Save a Cv and Ca for 2 seg and 1 align(copy code to alignment_asot.py)
# ## and make sure each of them has a unique name to prevent overwriting when saving
# ## Maybe pass a flag to the segment_asot() to indicate whether its for vid X or vid Y


#     # N is no. of frames in video X, K is no. of frames in video Y
#     MY_Nr = int(round(N * radius))
#     Kr = int(round(K * radius))

#     # Compute Cv
#     MY_Cv = np.zeros((N, N))
#     for i in range(N):
#         for k in range(N):
#             if 1 <= abs(i - k) <= MY_Nr:
#                 MY_Cv[i, k] = 1 / radius

#     # Compute Ca
#     MY_Ca = np.ones((K, K))  # Default to 1
#     for j in range(K):
#         for l in range(K):
#             if 1 <= abs(j - l) <= Kr:
#                 MY_Ca[j, l] = 0

#     # Save as .npy files
#     np.save(f"align_X_Cv.npy", MY_Cv)
#     np.save(f"align_Y_Ca.npy", MY_Ca)
# ############DE-BUG BUG BUG BUG########################################

    # the trace stores all of the computed objs(transportation costs)
    obj_trace = []
    # Iteration number for the outer loop
    it = 0

    while True:
        with torch.no_grad():
            # Compute the transportation cost for the current transportation matrix T, using the ASOT objective function
            obj = ot_objective(T, cost_matrix, eps, alpha, radius, ub_frames, ub_actions,
                                lambda_frames, lambda_actions, mask_X=mask_X, mask_Y=mask_Y)
        # Append the current obj to the trace
        obj_trace.append(obj)
        
        # Exit the loop when all outer iterations are completed
        if it >= n_iters[0]:
            break
        
        # TODO: Might have to add/remove gradient terms when we make any changes above this
        # gradient of objective function required for mirror descent step
        fgw_cost_matrix = grad_fgw(T, cost_matrix, alpha, Cv, Ca)
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
            T = project_to_polytope_KL(fgw_cost_matrix, mask_X, mask_Y, eps, dx, dy,
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

    # print(nnz_X.shape)
    # print(nnz_X)
    # The row-sum of every row will be 1.0
    # TODO: The col-sum of each col is not enforced to be 1.0 right now, might need to change this for VAOT

    T = T * nnz_X[:, None, None]  # rescale so marginals per frame(of video X) sum to 1
    # print(T)
    # row_sum = T.sum(dim=1)
    # is_row_sum_1 = (row_sum == 1).all()

    # print(is_row_sum_1)
    # print(T[:,1:,1:].shape)
    obj_trace = torch.cat(obj_trace)
    return T[:,1:,1:], obj_trace

# ρR = rho * Temporal prior
# In the temporal prior formula: N=no. of sampled frames from video X, K=no. of sampled frames from video Y (Used to be no. of clusters in ASOT)
def temporal_prior(n_frames_X, n_frames_Y, rho, device):
    temp_prior = torch.abs(torch.arange(n_frames_X)[:, None] / n_frames_X - torch.arange(n_frames_Y)[None, :] / n_frames_Y).to(device)
    return rho * temp_prior
