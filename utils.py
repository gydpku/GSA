import torch
def flip_inner(x, flip1, flip2):
    num = x.shape[0]

    # print(num)
    a = x  # .permute(0,1,3,2)
    a = a.view(num, 3, 2, 16, 32)
    #  imshow(torchvision.utils.make_grid(a))
    a = a.permute(2, 0, 1, 3, 4)
    s1 = a[0]  # .permute(1,0, 2, 3)#, 4)
    s2 = a[1]  # .permute(1,0, 2, 3)
    # print("a",a.shape,a[:63][0].shape)
    if flip1:
        s1 = torch.flip(s1, (3,))  # torch.rot90(s1, 2*rot1, (2, 3))
    if flip2:
        s2 = torch.flip(s2, (3,))  # torch.rot90(s2, 2*rot2, (2, 3))

    s = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2)
    # imshow(torchvision.utils.make_grid(s[2]))
    #   print("s",s.shape)
    # S = s.permute(0,1, 2, 3, 4)  # .view(3,32,32)
    # print("S",S.shape)
    S = s.reshape(num, 3, 32, 32)
    # S =S.permute(0,1,3,2)
    # imshow(torchvision.utils.make_grid(S[2]))
    #    print("S", S.shape)
    return S


def RandomFlip(x, num):
    # print(x.shape)
    #aug_x = simclr_aug(x)
    x=simclr_aug(x)
    X = []
    # print(x.shape)

    # for i in range(4):
    X.append(x)
    X.append(flip_inner(x, 1, 1))

    X.append(flip_inner(x, 0, 1))

    X.append(flip_inner(x, 1, 0))
    # else:
    #   x1=rot_inner(x,0,1)

    return torch.cat([X[i] for i in range(num)], dim=0)


def rot_inner(x, rot1, rot2):
    num = x.shape[0]

    # print(num)
    a = x.permute(0, 1, 3, 2)
    a = a.view(num, 3, 2, 16, 32)
    #  imshow(torchvision.utils.make_grid(a))
    a = a.permute(2, 0, 1, 3, 4)
    s1 = a[0]  # .permute(1,0, 2, 3)#, 4)
    s2 = a[1]  # .permute(1,0, 2, 3)
    # print("a",a.shape,a[:63][0].shape)
    s1 = torch.rot90(s1, 2 * rot1, (2, 3))
    s2 = torch.rot90(s2, 2 * rot2, (2, 3))

    s = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2)

    S = s.reshape(num, 3, 32, 32)
    S = S.permute(0, 1, 3, 2)

    return S


def Rotation(x, r):
    # print(x.shape)
    x = torch.rot90(x, r, (2, 3))
    X = []
    # print(x.shape)

    X.append(rot_inner(x, 0, 0))

    X.append(rot_inner(x, 1, 1))

    X.append(rot_inner(x, 1, 0))

    X.append(rot_inner(x, 0, 1))


    return x
import torch.nn.functional as F
def cutmix_data(x, y, Basic_model,alpha=1.0, cutmix_prob=0.5,):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam,x,Basic_model)
    #for ii in range(batch_size):x[ii,:,bbx1[ii]:bbx2[ii],bby1[ii]:bby2[ii]]=x[index][ii,:,bbx1[index][ii]:bbx2[index][ii],bby1[index][ii]:bby2[index][ii]]

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def rand_bbox(size, lam,x,Basic_model):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    feat = feat_normalized(Basic_model, x).reshape(-1,W,H)
    import pdb
    #pdb.set_trace()
   # cx=torch.mean(feat,dim=2).max(dim=1)[1].cpu()
   # cy=torch.mean(feat,dim=1).max(dim=1)[1].cpu()
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
def flip_inner(x, flip1, flip2):
    num = x.shape[0]


    # print(num)
    a = x  # .permute(0,1,3,2)
    a = a.view(num, 3, 2, 16, 32)
    #  imshow(torchvision.utils.make_grid(a))
    a = a.permute(2, 0, 1, 3, 4)
    s1 = a[0]  # .permute(1,0, 2, 3)#, 4)
    s2 = a[1]  # .permute(1,0, 2, 3)
    # print("a",a.shape,a[:63][0].shape)
    if flip1:
        s1 = torch.flip(s1, (3,))  # torch.rot90(s1, 2*rot1, (2, 3))
    if flip2:
        s2 = torch.flip(s2, (3,))  # torch.rot90(s2, 2*rot2, (2, 3))

    s = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2)
    # imshow(torchvision.utils.make_grid(s[2]))
    #   print("s",s.shape)
    # S = s.permute(0,1, 2, 3, 4)  # .view(3,32,32)
    # print("S",S.shape)
    S = s.reshape(num, 3, 32, 32)
    # S =S.permute(0,1,3,2)
    # imshow(torchvision.utils.make_grid(S[2]))
    #    print("S", S.shape)
    return S


def RandomFlip(x, num):
    # print(x.shape)
    #aug_x = simclr_aug(x)
   # x=simclr_aug(x)
    X = []
    # print(x.shape)

    # for i in range(4):
    X.append(simclr_aug(x))
    X.append(flip_inner(simclr_aug(x), 1, 1))

    X.append(flip_inner(x, 0, 1))

    X.append(flip_inner(x, 1, 0))
    # else:
    #   x1=rot_inner(x,0,1)

    return torch.cat([X[i] for i in range(num)], dim=0)


def rot_inner(x):
    num = x.shape[0]

    # print(num)
    R = x.repeat(4, 1, 1, 1)
    a = x.permute(0, 1, 3, 2)
    a = a.view(num, 3, 2, 16, 32)
    import pdb
    # pdb.set_trace()
    #  imshow(torchvision.utils.make_grid(a))
    a = a.permute(2, 0, 1, 3, 4)
    s1 = a[0]  # .permute(1,0, 2, 3)#, 4)
    s2 = a[1]  # .permute(1,0, 2, 3)
    a = torch.rot90(a, 2, (3, 4))
    s1_1 = a[0]  # .permute(1,0, 2, 3)#, 4)
    s2_2 = a[1]  # .permute(1,0, 2, 3)
    # S0 = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num, 1, 28, 28).permute(0, 1, 3, 2)
    R[3 * num:] = torch.cat((s1_1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1, 3, 2)
    R[num:2 * num] = torch.cat((s1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1, 3, 2)
    R[2 * num:3 * num] = torch.cat((s1_1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1,
                                                                                                                  3, 2)

    return R



def test_model_cur(loder, i):
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(loder):
        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        Basic_model.eval()
        pred = Basic_model.forward(data)[:,2*(i):2*(i+1)]
        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]
        target=target-2*i

        #    print("final", Pred, target.data.view_as(Pred))
        # print(target,"True",pred)

        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
    print(
        'Test set{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
            .format(i,
                    test_loss, correct, num,
                    100. * correct / num, ))
    return test_accuracy

def test_model_past(loder, i):
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(loder):
        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        Basic_model.eval()
        pred = Basic_model.forward(data)[:,:2*(i+1)]
        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]
        target=target

        #    print("final", Pred, target.data.view_as(Pred))
        # print(target,"True",pred)

        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
    print(
        'Test set{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
            .format(i,
                    test_loss, correct, num,
                    100. * correct / num, ))
    return test_accuracy
def test_model_future(loder, i):
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(loder):
        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        Basic_model.eval()
        pred = Basic_model.forward(data)[:,2*i:]
        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]
        target=target-2*i

        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
    print(
        'Test set{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
            .format(i,
                    test_loss, correct, num,
                    100. * correct / num, ))
    return test_accuracy


def test_model(loder, i):
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(loder):
        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        Basic_model.eval()
        pred = Basic_model.forward(data)
        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]

        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
    print(
        'Test set{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
            .format(i,
                    test_loss, correct, num,
                    100. * correct / num, ))
    return test_accuracy



def get_true_prob(x, y, llabel):
    num = x.size()[0]
    true = []
    true2 = []
    for i in range(num):

        if y[i] in llabel:
            true.append(1)
        else:
            true.append(0)
        # true.append(x[i][y[i]])
        # true2.append(0.5)

        # true.append(x[i][y[i]])
    return torch.FloatTensor(true).cuda()  # ,#torch.FloatTensor(true2).cuda()


def get_prob_rate(x, logits, label):
    num = x.size()[0]
    logits = F.softmax(logits, dim=1)
    rate = []
    # true2=[]
    for i in range(num):
        true_prob = logits[i][label[i]].item()
        max_prob = torch.max(logits[i])
        rate.append(true_prob / max_prob)
    return torch.FloatTensor(rate).cuda()


def get_prob_rate_cross( logits, label, t):
    logits = F.softmax(logits, dim=1)
    rate = []
    num = logits.size()[0]
    # true2=[]
    # import pdb
    # pdb.set_trace()
    for i in range(num):
        true_prob = logits[i][label[i]].item()
        # import pdb
        # pdb.set_trace()
        max_prob = torch.max(logits[i, :-t])
        rate.append(true_prob / max_prob)
    return torch.FloatTensor(rate).cuda()
def get_mean_rate_cross( logits, label, t):
    logits = F.softmax(logits, dim=1)
    rate = []
    num = logits.size()[0]
    # true2=[]
    # import pdb
    # pdb.set_trace()
    for i in range(num):
        true_prob = logits[i][label[i]].item()
        # import pdb
        # pdb.set_trace()
        max_prob = torch.max(logits[i, :-t])
        rate.append(true_prob / max_prob)
    return torch.FloatTensor(rate).cuda()

def square_diagonal_16(x):
    num = x.shape[0]

    # print(num)
    R = x.repeat(16, 1, 1, 1)
    uuu = x.unfold(2, 16, 16)
    vvv = uuu.unfold(3, 16, 16)
    vvv=vvv.reshape(-1,3,4,16,16)
    index1 = [0, 1,2,3]
    index2 = [0,1,3,2]
    index3 = [0,2,3,1]
    index4 = [0,2,1,3]# 2, 1, 3]
    index5 = [0,3, 1, 2]
    index6=[0,3,2,1]
    index7=[1,0,2,3]
    index8=[1,0,3,2]
    index9 = [1, 2, 3, 0]
    index10 = [1, 2, 0, 3]
    index11 = [1, 3, 2, 0]
    index12 = [1, 3, 0, 2]
    index13 = [2, 0, 1, 3]
    index14=[2,0,3,1]
    index15=[2,1,0,3]
    index_r = [1, 0]
    vvv1 = vvv[:, :, index1].reshape(-1,3,2,2,16,16)
    vvv2 = vvv[:, :, index2].reshape(-1,3,2,2,16,16)
    vvv3 = vvv[:, :, index3].reshape(-1,3,2,2,16,16)
    vvv4 = vvv[:, :, index4].reshape(-1, 3, 2, 2, 16, 16)
    vvv5 = vvv[:, :, index5].reshape(-1, 3, 2, 2, 16, 16)
    vvv6 = vvv[:, :, index6].reshape(-1, 3, 2, 2, 16, 16)
    vvv7 = vvv[:, :, index7].reshape(-1, 3, 2, 2, 16, 16)
    vvv8 = vvv[:, :, index8].reshape(-1, 3, 2, 2, 16, 16)
    vvv9 = vvv[:, :, index9].reshape(-1, 3, 2, 2, 16, 16)
    vvv10 = vvv[:, :, index10].reshape(-1, 3, 2, 2, 16, 16)
    vvv11 = vvv[:, :, index11].reshape(-1, 3, 2, 2, 16, 16)
    vvv12 = vvv[:, :, index12].reshape(-1, 3, 2, 2, 16, 16)
    vvv13 = vvv[:, :, index13].reshape(-1, 3, 2, 2, 16, 16)
    vvv14 = vvv[:, :, index14].reshape(-1, 3, 2, 2, 16, 16)
    vvv15 = vvv[:, :, index15].reshape(-1, 3, 2, 2, 16, 16)

    vvv1 = torch.cat((vvv1[:, :, 0].squeeze(2), vvv1[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv1 = torch.cat((vvv1[:, :, 0].squeeze(2), vvv1[:, :, 1].squeeze(2)), dim=3)
    vvv2 = torch.cat((vvv2[:, :, 0].squeeze(2), vvv2[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv2 = torch.cat((vvv2[:, :, 0].squeeze(2), vvv2[:, :, 1].squeeze(2)), dim=3)
    vvv3 = torch.cat((vvv3[:, :, 0].squeeze(2), vvv3[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv3 = torch.cat((vvv3[:, :, 0].squeeze(2), vvv3[:, :, 1].squeeze(2)), dim=3)
    vvv4 = torch.cat((vvv4[:, :, 0].squeeze(2), vvv4[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv4 = torch.cat((vvv4[:, :, 0].squeeze(2), vvv4[:, :, 1].squeeze(2)), dim=3)
    vvv5 = torch.cat((vvv5[:, :, 0].squeeze(2), vvv5[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv5 = torch.cat((vvv5[:, :, 0].squeeze(2), vvv5[:, :, 1].squeeze(2)), dim=3)
    vvv6 = torch.cat((vvv6[:, :, 0].squeeze(2), vvv6[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv6 = torch.cat((vvv6[:, :, 0].squeeze(2), vvv6[:, :, 1].squeeze(2)), dim=3)
    vvv7 = torch.cat((vvv7[:, :, 0].squeeze(2), vvv7[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv7 = torch.cat((vvv7[:, :, 0].squeeze(2), vvv7[:, :, 1].squeeze(2)), dim=3)
    vvv8 = torch.cat((vvv8[:, :, 0].squeeze(2), vvv8[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv8 = torch.cat((vvv8[:, :, 0].squeeze(2), vvv8[:, :, 1].squeeze(2)), dim=3)
    vvv9 = torch.cat((vvv9[:, :, 0].squeeze(2), vvv9[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv9 = torch.cat((vvv9[:, :, 0].squeeze(2), vvv9[:, :, 1].squeeze(2)), dim=3)
    vvv10 = torch.cat((vvv10[:, :, 0].squeeze(2), vvv10[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv10 = torch.cat((vvv10[:, :, 0].squeeze(2), vvv10[:, :, 1].squeeze(2)), dim=3)
    vvv11 = torch.cat((vvv11[:, :, 0].squeeze(2), vvv11[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv11 = torch.cat((vvv11[:, :, 0].squeeze(2), vvv11[:, :, 1].squeeze(2)), dim=3)
    vvv12 = torch.cat((vvv12[:, :, 0].squeeze(2), vvv12[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv12 = torch.cat((vvv12[:, :, 0].squeeze(2), vvv12[:, :, 1].squeeze(2)), dim=3)
    vvv13 = torch.cat((vvv13[:, :, 0].squeeze(2), vvv13[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv13 = torch.cat((vvv13[:, :, 0].squeeze(2), vvv13[:, :, 1].squeeze(2)), dim=3)
    vvv14 = torch.cat((vvv14[:, :, 0].squeeze(2), vvv14[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv14 = torch.cat((vvv14[:, :, 0].squeeze(2), vvv14[:, :, 1].squeeze(2)), dim=3)
    vvv15 = torch.cat((vvv15[:, :, 0].squeeze(2), vvv15[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv15 = torch.cat((vvv15[:, :, 0].squeeze(2), vvv15[:, :, 1].squeeze(2)), dim=3)




    import pdb
    '''
    uvi = square_diagonal(x)
    imshow(torchvision.utils.make_grid(uvi[0]))
    imshow(torchvision.utils.make_grid(uvi[10]))
    imshow(torchvision.utils.make_grid(uvi[20]))
    imshow(torchvision.utils.make_grid(uvi[30]))
    '''
    # S0 = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num, 1, 28, 28).permute(0, 1, 3, 2)
    R[3 * num:4*num] = vvv3#torch.cat((s1_1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1, 3, 2)
    R[num:2 * num] = vvv1#torch.cat((s1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1, 3, 2)
    R[2 * num:3 * num] = vvv2#torch.cat((s1_1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1,
    R[
    4 * num:5 * num] = vvv4  # torch.cat((s1_1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1, 3, 2)
    R[
    5*num:6 * num] = vvv5  # torch.cat((s1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1, 3, 2)
    R[6 * num:7 * num] = vvv6
    R[
    7 * num:8 * num] = vvv7  # torch.cat((s1_1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1, 3, 2)
    R[
    8 * num:9 * num] = vvv8  # torch.cat((s1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1, 3, 2)
    R[9 * num:10 * num] = vvv9
    R[
    10 * num:11 * num] = vvv10  # torch.cat((s1_1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1, 3, 2)
    R[
    11 * num:12 * num] = vvv11  # torch.cat((s1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1, 3, 2)
    R[12 * num:13 * num] = vvv12
    R[
    13 * num:14 * num] = vvv13  # torch.cat((s1_1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1, 3, 2)
    R[
    14 * num:15 * num] = vvv14  # torch.cat((s1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1, 3, 2)
    R[15 * num:16 * num] = vvv15
    #3, 2)
#312 78.7
#
    return R
def square_diagonal(x):
    num = x.shape[0]

    # print(num)
    R = x.repeat(4, 1, 1, 1)
    #a = x.permute(0, 1, 3, 2)
    #a = a.view(num, 3, 2, 16, 32)
    uuu = x.unfold(2, 16, 16)
    vvv = uuu.unfold(3, 16, 16)
    vvv=vvv.reshape(-1,3,4,16,16)
    index1 = [0, 2,1,3]
    index2 = [3,1,2,0]
    index3 = [3,2,1,0]
    index_r = [1, 0]
    vvv1 = vvv[:, :, index1].reshape(-1,3,2,2,16,16)
    vvv2 = vvv[:, :, index2].reshape(-1,3,2,2,16,16)
    vvv3 = vvv[:, :, index3].reshape(-1,3,2,2,16,16)
    #vvv1 = vvv[:, :, index_r]
    #vvv2 = vvv[:, :, :,index_r]
    #vvv3 = vvv1[:, :, :, index_r]
   # vvv2 = vvv3[:, :, index_r]
    vvv1 = torch.cat((vvv1[:, :, 0].squeeze(2), vvv1[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv1 = torch.cat((vvv1[:, :, 0].squeeze(2), vvv1[:, :, 1].squeeze(2)), dim=3)
    vvv2 = torch.cat((vvv2[:, :, 0].squeeze(2), vvv2[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv2 = torch.cat((vvv2[:, :, 0].squeeze(2), vvv2[:, :, 1].squeeze(2)), dim=3)
    vvv3 = torch.cat((vvv3[:, :, 0].squeeze(2), vvv3[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv3 = torch.cat((vvv3[:, :, 0].squeeze(2), vvv3[:, :, 1].squeeze(2)), dim=3)
    import pdb
    '''
    uvi = square_diagonal(x)
    imshow(torchvision.utils.make_grid(uvi[0]))
    imshow(torchvision.utils.make_grid(uvi[10]))
    imshow(torchvision.utils.make_grid(uvi[20]))
    imshow(torchvision.utils.make_grid(uvi[30]))
    '''
    # pdb.set_trace()
    #  imshow(torchvision.utils.make_grid(a))
  #  a = a.permute(2, 0, 1, 3, 4)
  #  s1 = a[0]  # .permute(1,0, 2, 3)#, 4)
   # s2 = a[1]  # .permute(1,0, 2, 3)
    #a = torch.rot90(a, 2, (3, 4))
    #s1_1 = a[0]  # .permute(1,0, 2, 3)#, 4)
    #s2_2 = a[1]  # .permute(1,0, 2, 3)
    # S0 = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num, 1, 28, 28).permute(0, 1, 3, 2)
    R[3 * num:] = vvv3#torch.cat((s1_1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1, 3, 2)
    R[num:2 * num] = vvv1#torch.cat((s1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1, 3, 2)
    R[2 * num:3 * num] = vvv2#torch.cat((s1_1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1,
                                                                                                                  #3, 2)
#312 78.7
#
    return R

def square_diagonal_repeat(x):
    num = x.shape[0]

    # print(num)
    R = x.repeat(4, 1, 1, 1)
    #a = x.permute(0, 1, 3, 2)
    #a = a.view(num, 3, 2, 16, 32)
    uuu = x.unfold(2, 16, 16)
    vvv = uuu.unfold(3, 16, 16)
    vvv=vvv.reshape(-1,3,4,16,16)
    index1 = [0, 0,0,0]
    index2 = [1,1,1,1]
    index3 = [2,2,2,2]
    index_r = [1, 0]
    vvv1 = vvv[:, :, index1].reshape(-1,3,2,2,16,16)
    vvv2 = vvv[:, :, index2].reshape(-1,3,2,2,16,16)
    vvv3 = vvv[:, :, index3].reshape(-1,3,2,2,16,16)
    #vvv1 = vvv[:, :, index_r]
    #vvv2 = vvv[:, :, :,index_r]
    #vvv3 = vvv1[:, :, :, index_r]
   # vvv2 = vvv3[:, :, index_r]
    vvv1 = torch.cat((vvv1[:, :, 0].squeeze(2), vvv1[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv1 = torch.cat((vvv1[:, :, 0].squeeze(2), vvv1[:, :, 1].squeeze(2)), dim=3)
    vvv2 = torch.cat((vvv2[:, :, 0].squeeze(2), vvv2[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv2 = torch.cat((vvv2[:, :, 0].squeeze(2), vvv2[:, :, 1].squeeze(2)), dim=3)
    vvv3 = torch.cat((vvv3[:, :, 0].squeeze(2), vvv3[:, :, 1].squeeze(2)), dim=3)  # vvv.reshape(-1,3,2,32,16)
    vvv3 = torch.cat((vvv3[:, :, 0].squeeze(2), vvv3[:, :, 1].squeeze(2)), dim=3)
    import pdb
    '''
    uvi = square_diagonal(x)
    imshow(torchvision.utils.make_grid(uvi[0]))
    imshow(torchvision.utils.make_grid(uvi[10]))
    imshow(torchvision.utils.make_grid(uvi[20]))
    imshow(torchvision.utils.make_grid(uvi[30]))
    '''
    # pdb.set_trace()
    #  imshow(torchvision.utils.make_grid(a))
  #  a = a.permute(2, 0, 1, 3, 4)
  #  s1 = a[0]  # .permute(1,0, 2, 3)#, 4)
   # s2 = a[1]  # .permute(1,0, 2, 3)
    #a = torch.rot90(a, 2, (3, 4))
    #s1_1 = a[0]  # .permute(1,0, 2, 3)#, 4)
    #s2_2 = a[1]  # .permute(1,0, 2, 3)
    # S0 = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num, 1, 28, 28).permute(0, 1, 3, 2)
    R[3 * num:] = vvv3#torch.cat((s1_1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1, 3, 2)
    R[num:2 * num] = vvv1#torch.cat((s1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1, 3, 2)
    R[2 * num:3 * num] = vvv2#torch.cat((s1_1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32).permute(0, 1,
                                                                                                                  #3, 2)
#312 78.7
#
    return R

def rot_inner_hlip(x):
    num = x.shape[0]

    # print(num)
    R = x.repeat(4, 1, 1, 1)
    a = x#.permute(0, 1, 3, 2)
    a = a.view(num, 3, 2, 16, 32)
    import pdb
    # pdb.set_trace()
    #  imshow(torchvision.utils.make_grid(a))
    a = a.permute(2, 0, 1, 3, 4)
    s1 = a[0]  # .permute(1,0, 2, 3)#, 4)
    s2 = a[1]  # .permute(1,0, 2, 3)
    a = torch.rot90(a, 2, (3, 4))
    s1_1 = a[0]  # .permute(1,0, 2, 3)#, 4)
    s2_2 = a[1]  # .permute(1,0, 2, 3)
    # S0 = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num, 1, 28, 28).permute(0, 1, 3, 2)
    R[3 * num:] = torch.cat((s1_1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32)#.permute(0, 1, 3, 2)
    R[num:2 * num] = torch.cat((s1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32)#.permute(0, 1, 3, 2)
    R[2 * num:3 * num] = torch.cat((s1_1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num, 3, 32, 32)#.permute(0, 1,
                                                                                                            #      3, 2)

    return R

def Rotation(x, oop):
    # print(x.shape)
    num = x.shape[0]
    X = square_diagonal(x)#rot_inner(x)  # , 1, 0)
   # X = rot_inner(X)
    X2=rot_inner(x)

    return torch.cat((X, torch.rot90(X, 1, (2, 3)), torch.rot90(X, 2, (2, 3)), torch.rot90(X, 3, (2, 3)),X2,torch.rot90(X2, 1, (2, 3))), dim=0)[
           :num * oop]
import matplotlib.pyplot as plt
import numpy as np
def imshow(img):
    img=img/2+0.5
    npimg=img.cpu().numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
def feat_normalized_hat(model,x,task_id):
    images = x.cuda(non_blocking=True)
    feat_map = model.f_train_feat_map(images,t=task_id,s=1)  # (N, C, H, W)
    N, Cf, Hf, Wf = feat_map.shape
    eval_train_map = feat_map.sum(1).view(N, -1)  # (N, Hf*Wf)
    eval_train_map = eval_train_map - eval_train_map.min(1, keepdim=True)[0]
    eval_train_map = eval_train_map / eval_train_map.max(1, keepdim=True)[0]
    eval_train_map = eval_train_map.view(N, 1, Hf, Wf)
    eval_train_map = F.interpolate(eval_train_map, size=images.shape[-2:], mode='bilinear')
    return eval_train_map
def feat_cam_normalized(model,x,y):
    images = x.cuda(non_blocking=True)
    feat_map = model.module.f_train_feat_map(images)  # (N, C, H, W)
    N, Cf, Hf, Wf = feat_map.shape
    #import pdb
    #pdb.set_trace()
    feat_map=torch.bmm(model.module.linear.weight[y].unsqueeze(1),feat_map.reshape(N,Cf,Hf*Wf))
    eval_train_map = feat_map.sum(1).view(N, -1)  # (N, Hf*Wf)
    eval_train_map = eval_train_map - eval_train_map.min(1, keepdim=True)[0]
    eval_train_map = eval_train_map / eval_train_map.max(1, keepdim=True)[0]
    eval_train_map = eval_train_map.view(N, 1, Hf, Wf)
    eval_train_map = F.interpolate(eval_train_map, size=images.shape[-2:], mode='bilinear')
    return eval_train_map
def feat_normalized(model,x):
    images = x.cuda(non_blocking=True)
    feat_map = model.f_train_feat_map(images)  # (N, C, H, W)
    N, Cf, Hf, Wf = feat_map.shape
    eval_train_map = feat_map.sum(1).view(N, -1)  # (N, Hf*Wf)
    eval_train_map = eval_train_map - eval_train_map.min(1, keepdim=True)[0]
    eval_train_map = eval_train_map / eval_train_map.max(1, keepdim=True)[0]
    eval_train_map = eval_train_map.view(N, 1, Hf, Wf)
    eval_train_map = F.interpolate(eval_train_map, size=images.shape[-2:], mode='bilinear')
    return eval_train_map
def Hbeta_torch(D, beta=1.0):
    P = torch.exp(-D.clone() * beta)

    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P


def x2p_torch(X, tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros(n, n)
    beta = torch.ones(n, 1)
    logU = torch.log(torch.tensor([perplexity]))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]

        (H, thisP) = Hbeta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta_torch(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P


def pca_torch(X, no_dims=50):
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - torch.mean(X, 0)

    (l, M) = torch.eig(torch.mm(X.t(), X), True)
    # split M real
    # this part may be some difference for complex eigenvalue
    # but complex eignevalue is meanless here, so they are replaced by their real part
    i = 0
    while i < d:
        if l[i, 1] != 0:
            M[:, i+1] = M[:, i]
            i += 2
        else:
            i += 1

    Y = torch.mm(X, M[:, 0:no_dims])
    return Y

def tsne(X, no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should not have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca_torch(X, initial_dims)
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, no_dims)
    dY = torch.zeros(n, no_dims)
    iY = torch.zeros(n, no_dims)
    gains = torch.ones(n, no_dims)

    # Compute P-values
    P = x2p_torch(X, 1e-5, perplexity)
    P = P + P.t()
    P = P / torch.sum(P)
    P = P * 4.    # early exaggeration
    print("get P shape", P.shape)
    P = torch.max(P, torch.tensor([1e-21]))

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor([1e-12]))

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(no_dims, 1).t() * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y
def test_model_conti(Basic_model,Loder,j):
    test_accuracy = 0
    task_num=len(Loder)
    for kk in range(len(Loder)):
        k=j
        correct = 0
        num = 0
        for batch_idx, (data, target) in enumerate(Loder):
            data, target = data.cuda(), target.cuda()
            # data, target = Variable(data, volatile=True), Variable(target)
            Basic_model.eval()
            mask=torch.nn.functional.one_hot(target%10,num_classes=10)
           # pdb.set_trace()
            pred = Basic_model.forward(data)#[:,:10*task_num]#torch.cat((Basic_model.forward(data)[:,10*(i):10*(i+1)]*mask,Basic_model.forward(data)[:,10*(j):10*(j+1)]),dim=1)
            pred[:,10*k:10*(k+1)]=pred[:,10*k:10*(k+1)]*mask
            Pred = pred.data.max(1, keepdim=True)[1]
            num += data.size()[0]
            target=target

            #    print("final", Pred, target.data.view_as(Pred))
            # print(target,"True",pred)

            correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

        test_accuracy += (100. * correct / num)#*0.5  # len(data_loader.dataset)

 #   print(
  #      'Test set{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
   #         .format(i,
    #                test_loss, correct, num,
     #               100. * correct / num, ))
    return test_accuracy/task_num
def test_model_task(Basic_model,loder1,loder2, i,j):
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(loder1):
        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        Basic_model.eval()
        mask=torch.nn.functional.one_hot(target%10,num_classes=10)
       # pdb.set_trace()
        pred = torch.cat((Basic_model.forward(data)[:,10*(i):10*(i+1)]*mask,Basic_model.forward(data)[:,10*(j):10*(j+1)]),dim=1)

        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]
        target=target-10*i

        #    print("final", Pred, target.data.view_as(Pred))
        # print(target,"True",pred)

        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = (100. * correct / num)*0.5  # len(data_loader.dataset)
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(loder2):
        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        Basic_model.eval()
        mask = torch.nn.functional.one_hot(target % 10, num_classes=10)
        #pdb.set_trace()
        pred = torch.cat((Basic_model.forward(data)[:, 10 * (i):10 * (i + 1)],
                          Basic_model.forward(data)[:, 10 * (j):10 * (j + 1)]* mask),dim=1)

        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]
        target = target - 10 * j +10

        #    print("final", Pred, target.data.view_as(Pred))
        # print(target,"True",pred)

        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy += (100. * correct / num)*0.5
 #   print(
  #      'Test set{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
   #         .format(i,
    #                test_loss, correct, num,
     #               100. * correct / num, ))
    return test_accuracy

def test_model_cur(Basic_model,loder, i):
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(loder):
        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        Basic_model.eval()
        pred = Basic_model.forward(data)[:,10*(i):10*(i+1)]
        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]
        target=target-10*i

        #    print("final", Pred, target.data.view_as(Pred))
        # print(target,"True",pred)

        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
 #   print(
  #      'Test set{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
   #         .format(i,
    #                test_loss, correct, num,
     #               100. * correct / num, ))
    return test_accuracy

def test_model_past(Basic_model,loder, i):
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(loder):
        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        Basic_model.eval()
        pred = Basic_model.forward(data)[:,:10*(i+1)]
        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]
        target=target

        #    print("final", Pred, target.data.view_as(Pred))
        # print(target,"True",pred)

        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
  #  print(
   #     'Test set{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
    #        .format(i,
     #               test_loss, correct, num,
      #              100. * correct / num, ))
    return test_accuracy

def test_model_mix(Basic_model,loder, i):
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(loder):
        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        Basic_model.eval()
        pred = torch.cat((Basic_model.forward(data)[:,10*(i):10*(i+1)],Basic_model.forward(data)[:,-10:]),dim=1)
        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]
        target=target-10*i

        #    print("final", Pred, target.data.view_as(Pred))
        # print(target,"True",pred)

        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
 #   print(
  #      'Test set{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
   #         .format(i,
    #                test_loss, correct, num,
     #               100. * correct / num, ))
    return test_accuracy
def test_model_future(Basic_model,loder, i):
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(loder):
        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        Basic_model.eval()
        pred = Basic_model.forward(data)[:,10*i:]
        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]
        target=target-10*i

        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
    print(
        'Test set{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
            .format(i,
                    test_loss, correct, num,
                    100. * correct / num, ))
    return test_accuracy


def test_model(Basic_model,loder, i):
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(loder):
        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        Basic_model.eval()
        pred = Basic_model.forward(data)
        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]

        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
    print(
        'Test set{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
            .format(i,
                    test_loss, correct, num,
                    100. * correct / num, ))
    return test_accuracy



def get_true_prob(x, y, llabel):
    num = x.size()[0]
    true = []
    true2 = []
    for i in range(num):

        if y[i] in llabel:
            true.append(1)
        else:
            true.append(0)
        # true.append(x[i][y[i]])
        # true2.append(0.5)

        # true.append(x[i][y[i]])
    return torch.FloatTensor(true).cuda()  # ,#torch.FloatTensor(true2).cuda()


def get_prob_rate(x, logits, label):
    num = x.size()[0]
    logits = F.softmax(logits, dim=1)
    rate = []
    # true2=[]
    for i in range(num):
        true_prob = logits[i][label[i]].item()
        max_prob = torch.max(logits[i])
        rate.append(true_prob / max_prob)
    return torch.FloatTensor(rate).cuda()


def get_prob_rate_cross( logits, label, t):
    logits = F.softmax(logits, dim=1)
    rate = []
    num = logits.size()[0]
    # true2=[]
    # import pdb
    # pdb.set_trace()
    for i in range(num):
        true_prob = logits[i][label[i]].item()
        # import pdb
        # pdb.set_trace()
        max_prob = torch.max(logits[i, :-t])
        rate.append(true_prob / max_prob)
    return torch.FloatTensor(rate).cuda()
def get_mean_rate_cross( logits, label, t):
    logits = F.softmax(logits, dim=1)
    rate = []
    num = logits.size()[0]
    # true2=[]
    # import pdb
    # pdb.set_trace()
    for i in range(num):
        true_prob = logits[i][label[i]].item()
        # import pdb
        # pdb.set_trace()
        max_prob = torch.max(logits[i, :-t])
        rate.append(true_prob / max_prob)
    return torch.FloatTensor(rate).cuda()
