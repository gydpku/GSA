import torch
import numpy

def generalized_contrastive_loss(
    hidden1,
    hidden2,
    lambda_weight=0.5,
    temperature=0.5,
    dist='normal',
    hidden_norm=True,
    loss_scaling=2.0):
  """Generalized contrastive loss.

  Both hidden1 and hidden2 should have shape of (n, d).

  Configurations to get following losses:
  * decoupled NT-Xent loss: set dist='logsumexp', hidden_norm=True
  * SWD with normal distribution: set dist='normal', hidden_norm=False
  * SWD with uniform hypersphere: set dist='normal', hidden_norm=True
  * SWD with uniform hypercube: set dist='uniform', hidden_norm=False
  """
  hidden_dim = hidden1.shape[-1]  # get hidden dimension
  #print(hidden_dim)
  #print(hidden1.shape)
  if hidden_norm:
    hidden1 = hidden1 / (hidden1.norm(dim=1, keepdim=True) + 1e-8)#torchtf.math.l2_normalize(hidden1, -1)
    hidden2 = hidden2 / (hidden2.norm(dim=1, keepdim=True) + 1e-8)
  loss_align = torch.mean((hidden1 - hidden2)**2)/2
  #print(loss_align)
  hiddens = torch.cat([hidden1, hidden2], 0)
  #print(hiddens.shape)
  if dist == 'logsumexp':
    loss_dist_match = get_logsumexp_loss(hiddens, temperature)
  else:
    a = torch.empty([hidden_dim, hidden_dim]).normal_(0, 1).cuda()
    rand_w = torch.nn.init.orthogonal_(a).cuda()
    #print("a",a==rand_w)
    #rand_w=a
   # print("send",rand_w)
   # print("rand",rand_w.shape)
    #initializer = torch.nn.init.orthogonal()# tf.keras.initializers.Orthogonal()
    #rand_w = initializer([hidden_dim, hidden_dim])
    loss_dist_match = get_swd_loss(hiddens, rand_w,
                            prior=dist,
                            hidden_norm=hidden_norm)
  a= loss_scaling * (-loss_align + lambda_weight * loss_dist_match)

  #print("a",loss_dist_match)
  return a,loss_align,loss_dist_match



def get_logsumexp_loss(states, temperature):
  scores = torch.matmul(states, states.t()) .cuda() # (bsz, bsz)
  bias = torch.log(torch.tensor(states.shape[1]).float()).cuda()
  #print(bias)
 # eye = torch.eye(scores.shape[1]).cuda()# a constant
  return  torch.mean(torch.log(torch.sum(torch.exp(scores / temperature),dim=1)+1e-8).cuda()).cuda()


def sort(x):
  """Returns the matrix x where each row is sorted (ascending)."""
  u = x.detach().cpu().numpy()

  t = numpy.argsort(u, axis=1)
  p = torch.from_numpy(t).long().cuda()
  b = torch.gather(x, -1, p)
  return b
  ''' 
  xshape = x.shape
  print(xshape[1])
  rank = torch.sum((x.unsqueeze(2) > x.unsqueeze(1)), dim=2).cuda()
  print("r",rank)
  for i in range(128):
    for j in range(128):
      if rank[i][j] < 0:
        print(rank[i][j])
      elif rank[i][j] >= 128:
        print("r",rank[i][j])
  rank_inv = torch.einsum(
    'dbc,c->db',
    torch.Tensor.permute(torch.nn.functional.one_hot(rank.long(), xshape[1]), [0, 2, 1]).float().cuda(),
    torch.arange(xshape[1]).float().cuda()).cuda() # (dim, bsz)
     # x = gather_nd(x, rank_inv.int(), axis=-1, batch_dims=-1)
  q= torch.nn.functional.one_hot(rank, xshape[1]).transpose(2,1).float().cpu()
  print("a")
  #q=torch.from_numpy(numpy.transpose(torch.nn.functional.one_hot(rank, xshape[1]).int().cpu().numpy(), [0, 2, 1])).float().cuda()
  for i in range(128):
    print(torch.sum(q[31][i]),i)
  print(q.shape)
  print(torch.sum(q[31][60]))
  t = torch.matmul(q[31][60], torch.arange(xshape[1]).float())
  print(t)
  t = numpy.array(t.cpu())
  q = numpy.array(q.cpu())
  #numpy.savetxt('/home/guoyd/Dataset/np2.txt', t)
  numpy.savetxt('/home/guoyd/Dataset/np.txt', q[31][60])
 # t=torch.matmul(q[31],torch.arange(xshape[1]).float().cuda())
   #              torch.arange(xshape[1]).float().cuda().cuda())
  #print("rr",q==rank_inv)
  #l=[]
  # w=False
  # s=0
  for i in range(128):
    for j in range(128):
      if rank_inv[i][j]<0:
        print(rank_inv[i][j])
      elif rank_inv[i][j]>=128:
        print(rank_inv[i][j],i,j)
        w=True
        s=i
        #for s in range(128):
         # print(rank_inv[31][s])
  #if w:
   # for j in range(128):
    #  l.append(rank_inv[s][j])
    #l=l.sort()
    #for i in range(128):
     # print(l[i])
  p=list(rank_inv[s][:])
  p.sort()
  n=0
  for i in range(len(p)):
    print(p[i],len(p),n)
    n=n+1

  #print(rank_inv[i][s])
  b = torch.gather(x, -1, rank_inv.long().cuda())
  #print("b",b)
  '''
 # return b




def get_swd_loss(states, rand_w, prior='normal', stddev=1., hidden_norm=True):
  states_shape = states.shape
  #print("get", rand_w)
  states = torch.matmul(states, rand_w)
  #print("get", rand_w)
  states_t = sort(states.t())
  #print("get2",states_t)# (dim, bsz)
  #print("get", rand_w)
  #print("t",states_t)
  #print("p",prior)
  if prior == 'normal':
    states_prior = torch.empty(states_shape).normal_(mean=1e-6,std=1+1e-8)#torch.randn(states_shape, mean=0, stddev=stddev)
  elif prior == 'uniform':
    states_prior = torch.empty(states_shape).uniform_(-1.0,1.0)
  else:
    raise ValueError('Unknown prior {}'.format(prior))
  #print("s", states_prior)
  if hidden_norm:
    states_prior = states_prior / (states_prior.norm(dim=1, keepdim=True) + 1e-8)
    #tf.math.l2_normalize(states_prior, -1)
  #print("get", rand_w)
  states_prior = torch.matmul(states_prior.cuda(), rand_w)
 # print("S", states_prior)
  states_prior_t = sort(states_prior.t())  # (dim, bsz)
  #print("ss",states_prior_t)
  #a=torch.mean((states_prior_t - states_t)**2)
  #print("los",states_prior_t-states_t)
  return torch.mean((states_prior_t - states_t)**2)


''' 
def get_contrastive_loss(z1, z2, nt_xent_temp):   # [batch_size, dim]
  batch_size = tf.shape(z1)[0]
  dim = tf.shape(z1)[1]

  z1 = tf.math.l2_normalize(z1, -1)
  z2 = tf.math.l2_normalize(z2, -1)

  sim = tf.matmul(z1, z2, transpose_b=True)  # [batch_size, batch_size]
  sim /= nt_xent_temp

  labels = tf.eye(batch_size)

  loss = (
      get_cls_loss(labels, sim) +
      get_cls_loss(labels, tf.transpose(sim))
  )
  return tf.reduce_mean(loss), sim

def get_cls_loss(labels, outputs):
  return tf.reduce_mean(cls_loss_object(labels, outputs))


cls_loss_object = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True,
    reduction=tf.keras.losses.Reduction.NONE)
'''
