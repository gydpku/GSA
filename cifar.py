import os,sys
import numpy as np
import torch
#import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle
import  torch.utils.data as Data


def get(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    t_num=2
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar_/'):
        os.makedirs('./data/binary_cifar_')
        t_num = 2
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        dat['train']=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR10('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(10//t_num):
            data[t] = {}
            data[t]['name'] = 'cifar10-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
            data[t]['ncla'] = t_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_num*t, t_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 10 // t_num
        data[t] = {}
        data[t]['name'] = 'cifar10-all'
        data[t]['ncla'] = 10
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar_'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar_'), 'data' + str(t) + s + 'y.bin'))

    # Load binary files
    data = {}
    ids = list(np.arange(6))
    print('Task order =', ids)
    for i in range(6):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar_'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar_'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar10->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n=0
    for t in data.keys():
        print("T",t)
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla'] = n
    Loder={}
    Loder_test={}
    for t in range(5):
        print("t",t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        Loder_test[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        u2 = torch.tensor(data[t]['test']['x'].reshape(-1, 3, 32, 32))
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        #u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=10,
            shuffle=True,
            )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
        #Loder[t]['valid'] = valid_loader
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    test_dataset= datasets.CIFAR10('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))#Data.TensorDataset(data[10//t_num]['test']['x'], data[10//t_num]['test']['y'])
    ''' 
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_dataset=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )
    Loder={}
    Loder[0] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
    Loder[0]['train']=train_loader
    '''
    test_dataset=datasets.CIFAR10('./data/', train=False, download=True,
                     transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10//data[0]['ncla']], size,Loder,test_loader
def get_pretrain_AOP(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    t_num=2
    # CIFAR10
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import pdb
    # pdb.set_trace()
    model, preprocess = clip.load('ViT-B/32', device)
    if not os.path.isdir('./data/binary_cifar_pretr/'):
        os.makedirs('./data/binary_cifar_pretr')
        t_num = 2
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        dat['train']=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR10('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(10//t_num):
            print(t,"t")
            num=0
            data[t] = {}
            data[t]['name'] = 'cifar10-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
            data[t]['ncla'] = t_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_num*t, t_num*(t+1)):
                        num+=1
                        if num%100==0:
                            print(num)
                     #   import pdb
                      #  pdb.set_trace()
                        with torch.no_grad():
                            image=transforms.ToPILImage()(image.squeeze(0))
                            image_input = preprocess(image).unsqueeze(0).to(device)
                            image_features = model.encode_image(image_input)
                            image=image_features.squeeze(0)
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 10 // t_num
        data[t] = {}
        data[t]['name'] = 'cifar10-all'
        data[t]['ncla'] = 10
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                with torch.no_grad():
                    image = transforms.ToPILImage()(image.squeeze(0))
                    image_input = preprocess(image).unsqueeze(0).to(device)
                    image_features = model.encode_image(image_input)
                    image = image_features.squeeze(0)
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, 512)
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar_p'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar_p'), 'data' + str(t) + s + 'y.bin'))

    # Load binary files
    data = {}
    ids = list(np.arange(6))
    print('Task order =', ids)
    for i in range(6):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar_p'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar_p'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar10->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    # pdb.set_trace()

    return data, taskcla[:10 // data[0]['ncla']], size
def get_pretrain(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    t_num=2
    # CIFAR10
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import pdb
    # pdb.set_trace()
    model, preprocess = clip.load('ViT-B/32', device)
    if not os.path.isdir('./data/binary_cifar_pretr/'):
        os.makedirs('./data/binary_cifar_pretr')
        t_num = 2
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        dat['train']=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR10('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(10//t_num):
            print(t,"t")
            num=0
            data[t] = {}
            data[t]['name'] = 'cifar10-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
            data[t]['ncla'] = t_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_num*t, t_num*(t+1)):
                        num+=1
                        if num%100==0:
                            print(num)
                     #   import pdb
                      #  pdb.set_trace()
                        with torch.no_grad():
                            image=transforms.ToPILImage()(image.squeeze(0))
                            image_input = preprocess(image).unsqueeze(0).to(device)
                            image_features = model.encode_image(image_input)
                            image=image_features.squeeze(0)
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 10 // t_num
        data[t] = {}
        data[t]['name'] = 'cifar10-all'
        data[t]['ncla'] = 10
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                with torch.no_grad():
                    image = transforms.ToPILImage()(image.squeeze(0))
                    image_input = preprocess(image).unsqueeze(0).to(device)
                    image_features = model.encode_image(image_input)
                    image = image_features.squeeze(0)
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, 512)
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar_p'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar_p'), 'data' + str(t) + s + 'y.bin'))

    # Load binary files
    data = {}
    ids = list(np.arange(6))
    print('Task order =', ids)
    for i in range(6):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar_p'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar_p'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar10->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n=0
    for t in data.keys():
        print("T",t)
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla'] = n
    Loder={}
    Loder_test={}
    for t in range(5):
        print("t",t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        Loder_test[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 512))  # .item()
        u2 = torch.tensor(data[t]['test']['x'].reshape(-1, 512))
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        #u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=32,
            shuffle=True,
            )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=32,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
        #Loder[t]['valid'] = valid_loader
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    test_dataset= datasets.CIFAR10('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))#Data.TensorDataset(data[10//t_num]['test']['x'], data[10//t_num]['test']['y'])
    ''' 
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_dataset=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )
    Loder={}
    Loder[0] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
    Loder[0]['train']=train_loader
    '''
    test_dataset=datasets.CIFAR10('./data/', train=False, download=True,transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
    #import pdb
    #pdb.set_trace()
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10//data[0]['ncla']], size,Loder,test_loader
def get_a_order(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    t_num=2
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar_a1/'):
        os.makedirs('./data/binary_cifar_a1')
        t_num = 2
        np.random.seed(101)
        cls_list = [i for i in range(10)]
        np.random.shuffle(cls_list)
        class_mapping = np.array(cls_list, copy=True)
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        dat['train']=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR10('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(10//t_num):
            data[t] = {}
            data[t]['name'] = 'cifar10-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
            data[t]['ncla'] = t_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if cls_list.index(label) in range(t_num*t, t_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(cls_list.index(label))
        t = 10 // t_num
        data[t] = {}
        data[t]['name'] = 'cifar10-all'
        data[t]['ncla'] = 10
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(cls_list.index(label))

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar_a1'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar_a1'), 'data' + str(t) + s + 'y.bin'))

    # Load binary files
    data = {}
    ids = list(np.arange(6))
    print('Task order =', ids)
    for i in range(6):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar_a1'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar_a1'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar10->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n=0
    for t in data.keys():
        print("T",t)
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla'] = n
    Loder={}
    Loder_test={}
    for t in range(5):
        print("t",t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        Loder_test[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        u2 = torch.tensor(data[t]['test']['x'].reshape(-1, 3, 32, 32))
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        #u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=32,
            shuffle=True,
            )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
        #Loder[t]['valid'] = valid_loader
   # mean = [x / 255 for x in [125.3, 123.0, 113.9]]
   # std = [x / 255 for x in [63.0, 62.1, 66.7]]
   # test_dataset= datasets.CIFAR10('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))#Data.TensorDataset(data[10//t_num]['test']['x'], data[10//t_num]['test']['y'])
   # test_dataset=datasets.CIFAR10('./data/', train=False, download=True,
    #                 transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
    dataset_new_test = Data.TensorDataset(data[5]['test']['x'], data[5]['test']['y'])

    test_loader = torch.utils.data.DataLoader(
        dataset_new_test,
        batch_size=64,
        shuffle=True,
    )
    #test_loader = torch.utils.data.DataLoader(
     #   test_dataset,
     #   batch_size=64,
     #   shuffle=True,
    #)
    print("Loder is prepared")
    return data, taskcla[:10//data[0]['ncla']], size,Loder,test_loader

def get_revisit(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    t_num=2
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar_/'):
        os.makedirs('./data/binary_cifar_')
        t_num = 2
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        dat['train']=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR10('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(10//t_num):
            data[t] = {}
            data[t]['name'] = 'cifar10-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
            data[t]['ncla'] = t_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_num*t, t_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 10 // t_num
        data[t] = {}
        data[t]['name'] = 'cifar10-all'
        data[t]['ncla'] = 10
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar_'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar_'), 'data' + str(t) + s + 'y.bin'))

    # Load binary files
    data = {}
    ids = list(np.arange(6))
    print('Task order =', ids)
    for i in range(6):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar_'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar_'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar10->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n=0
    for t in data.keys():
        print("T",t)
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla'] = n
    Loder={}
    for t in range(5):
        print("t",t)

        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        #u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        for i in range(2):
            dataset_new_train = Data.TensorDataset(data[t]['train']['x'][i*int(TOTAL_NUM/2):(i+1)*int(TOTAL_NUM/2)], data[t]['train']['y'][i*int(TOTAL_NUM/2):(i+1)*int(TOTAL_NUM/2)])
        #dataset_new_valid = Data.TensorDataset(data[t]['valid']['x'], data[t]['valid']['y'])
            train_loader = torch.utils.data.DataLoader(
                dataset_new_train,
                batch_size=10,
                shuffle=True,
                )
            Loder[2 * t+ i] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
            Loder[2*t+i]['train'] = train_loader
        #Loder[t]['valid'] = valid_loader
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    test_dataset= datasets.CIFAR10('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))#Data.TensorDataset(data[10//t_num]['test']['x'], data[10//t_num]['test']['y'])
    ''' 
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_dataset=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )
    Loder={}
    Loder[0] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
    Loder[0]['train']=train_loader
    '''
    test_dataset=datasets.CIFAR10('./data/', train=False, download=True,
                     transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10//data[0]['ncla']], size,Loder,test_loader

def get_cifar100(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar100_100/'):
        os.makedirs('./data/binary_cifar100_100')
        t_num = 10
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        dat['train']=datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(100//t_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'cifar100-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
            data[t]['ncla'] = t_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_num*t, t_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 100 // t_num
        data[t] = {}
        data[t]['name'] = 'cifar100-all'
        data[t]['ncla'] = 100
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_100'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_100'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(11))
    print('Task order =', ids)
    for i in range(11):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar100_100'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar100_100'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(10):
        print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=10,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
  #  test_dataset = datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose(
   #     [transforms.ToTensor(), transforms.Normalize(mean,
    #                                                 std)]))  # Data.TensorDataset(data[10//t_num]['test']['x'], data[10//t_num]['test']['y'])
    ''' 
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_dataset=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )
    Loder={}
    Loder[0] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
    Loder[0]['train']=train_loader
    '''
    test_dataset = datasets.CIFAR100('./data/', train=False, download=True,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), transforms.Normalize(mean, std)]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader
def get_cifar100_joint(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar100_j/'):
        os.makedirs('./data/binary_cifar100_j')
        t_num = 100
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        dat['train']=datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(100//t_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'cifar100-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
            data[t]['ncla'] = t_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_num*t, t_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 100 // t_num
        data[t] = {}
        data[t]['name'] = 'cifar100-all'
        data[t]['ncla'] = 100
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_j'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_j'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(2))
    print('Task order =', ids)
    for i in range(2):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar100_j'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar100_j'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(1):
        print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=10,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
  #  test_dataset = datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose(
   #     [transforms.ToTensor(), transforms.Normalize(mean,
    #                                                 std)]))  # Data.TensorDataset(data[10//t_num]['test']['x'], data[10//t_num]['test']['y'])
    ''' 
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_dataset=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )
    Loder={}
    Loder[0] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
    Loder[0]['train']=train_loader
    '''
    test_dataset = datasets.CIFAR100('./data/', train=False, download=True,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), transforms.Normalize(mean, std)]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader
def get_cifar100_50(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar100_22/'):
        os.makedirs('./data/binary_cifar100_22')
        t_class_num = 2
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        dat['train']=datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(100//t_class_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'cifar100-' + str(t_class_num*t) + '-' + str(t_class_num*(t+1)-1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_class_num*t, t_class_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 100 // t_class_num
        data[t] = {}
        data[t]['name'] = 'cifar100-all'
        data[t]['ncla'] = 100
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_2'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_2'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(51))
    print('Task order =', ids)
    for i in range(51):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar100_2'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar100_2'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(50):
        print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=10,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
  #  test_dataset = datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose(
   #     [transforms.ToTensor(), transforms.Normalize(mean,
    #                                                 std)]))  # Data.TensorDataset(data[10//t_num]['test']['x'], data[10//t_num]['test']['y'])
    ''' 
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_dataset=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )
    Loder={}
    Loder[0] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
    Loder[0]['train']=train_loader
    '''
    test_dataset = datasets.CIFAR100('./data/', train=False, download=True,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), transforms.Normalize(mean, std)]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader
def get_mnist(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [1, 28, 28]
    # CIFAR10
    if not os.path.isdir('./data/binary_mnist/'):
        os.makedirs('./data/binary_mnist')
        t_class_num = 2
        mean = (0.1307,)
        std = (0.3081,)
        dat={}
        dat['train']=datasets.MNIST('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.MNIST('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(10//t_class_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'mnist' + str(t_class_num*t) + '-' + str(t_class_num*(t+1)-1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_class_num*t, t_class_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 10 // t_class_num
        data[t] = {}
        data[t]['name'] = 'mnist-all'
        data[t]['ncla'] = 10
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_mnist'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_mnist'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(6))
    print('Task order =', ids)
    for i in range(6):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_mnist'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_mnist'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'mnist->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(5):
        print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 1, 28, 28))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=10,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
    mean = (0.1307,)
    std = (0.3081,)

    test_dataset = datasets.MNIST('./data/', train=False, download=True,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), transforms.Normalize(mean, std)]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader
def get_cifar100_20(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar100_5/'):
        os.makedirs('./data/binary_cifar100_5')
        t_class_num = 5
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        dat['train']=datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(100//t_class_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'cifar100-' + str(t_class_num*t) + '-' + str(t_class_num*(t+1)-1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_class_num*t, t_class_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 100 // t_class_num
        data[t] = {}
        data[t]['name'] = 'cifar100-all'
        data[t]['ncla'] = 100
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_5'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_5'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(21))
    print('Task order =', ids)
    for i in range(21):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar100_5'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar100_5'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(20):
        print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=32,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
  #  test_dataset = datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose(
   #     [transforms.ToTensor(), transforms.Normalize(mean,
    #                                                 std)]))  # Data.TensorDataset(data[10//t_num]['test']['x'], data[10//t_num]['test']['y'])
    ''' 
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_dataset=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )
    Loder={}
    Loder[0] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
    Loder[0]['train']=train_loader
    '''
    test_dataset = datasets.CIFAR100('./data/', train=False, download=True,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), transforms.Normalize(mean, std)]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader
def get_cifar100_10(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar100_10/'):
        os.makedirs('./data/binary_cifar100_10')
        t_class_num = 10
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        dat['train']=datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(100//t_class_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'cifar100-' + str(t_class_num*t) + '-' + str(t_class_num*(t+1)-1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_class_num*t, t_class_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 100 // t_class_num
        data[t] = {}
        data[t]['name'] = 'cifar100-all'
        data[t]['ncla'] = 100
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_10'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_10'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(11))
    print('Task order =', ids)
    for i in range(11):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar100_10'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar100_10'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(10):
        print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=10,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
  #  test_dataset = datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose(
   #     [transforms.ToTensor(), transforms.Normalize(mean,
    #                                                 std)]))  # Data.TensorDataset(data[10//t_num]['test']['x'], data[10//t_num]['test']['y'])
    ''' 
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_dataset=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )
    Loder={}
    Loder[0] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
    Loder[0]['train']=train_loader
    '''
    test_dataset = datasets.CIFAR100('./data/', train=False, download=True,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), transforms.Normalize(mean, std)]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader
def get_cifar100_5_5(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar100_5_5/'):
        os.makedirs('./data/binary_cifar100_5_5')
        t_num = 9
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        dat['train']=datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(9):
            print(t)
            if t==0:
                data[t] = {}
                data[t]['name'] = 'cifar100-' + str(t_num * t) + '-' + str(t_num * (t + 1) - 1)
                data[t]['ncla'] = t_num
                for s in ['train', 'test']:
                    loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                    data[t][s] = {'x': [], 'y': []}
                    for image, target in loader:
                        label = target.numpy()[0]
                        if label in range(0, 60):
                            data[t][s]['x'].append(image)
                            data[t][s]['y'].append(label)
            else:
                data[t] = {}
                data[t]['name'] = 'cifar100-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
                data[t]['ncla'] = t_num
                for s in ['train', 'test']:
                    loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                    data[t][s] = {'x': [], 'y': []}
                    class_num={}
                    for i in range(60+5*(t-1),60+5*t):
                        class_num[i]=0
                    for image, target in loader:
                        label = target.numpy()[0]
                        if label in range(60+5*(t-1), 60+5*t):
                            if class_num[label]<5:
                                data[t][s]['x'].append(image)
                                data[t][s]['y'].append(label)
                                class_num[label]+=1
                            else:
                                continue
        t = 100 // t_num
        data[t] = {}
        data[t]['name'] = 'cifar100-all'
        data[t]['ncla'] = 100
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_5_5'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_5_5'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(9))
    print('Task order =', ids)
    for i in range(9):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar100_5_5'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar100_5_5'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(9):
        print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        # dataset_new_valid = Data.TensorDataset(data[t]['valid']['x'], data[t]['valid']['y'])
        if t==0:
            train_loader = torch.utils.data.DataLoader(
                dataset_new_train,
                batch_size=128,
                shuffle=True,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset_new_train,
                batch_size=25,
                shuffle=True,
            )

        Loder[t]['train'] = train_loader
        # Loder[t]['valid'] = valid_loader
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
  #  test_dataset = datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose(
   #     [transforms.ToTensor(), transforms.Normalize(mean,
    #                                                 std)]))  # Data.TensorDataset(data[10//t_num]['test']['x'], data[10//t_num]['test']['y'])
    ''' 
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_dataset=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )
    Loder={}
    Loder[0] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
    Loder[0]['train']=train_loader
    '''
    test_dataset = datasets.CIFAR100('./data/', train=False, download=True,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), transforms.Normalize(mean, std)]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2000,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader

from tinyimagenet import MyTinyImagenet
from conf import base_path
def get_tinyimagenet_100(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 64, 64]
    # CIFAR10
    if not os.path.isdir('./data/binary_tiny200_222/'):
        os.makedirs('./data/binary_tiny200_222')
        t_class_num = 2
        #mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        #std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transform])

        train = MyTinyImagenet(base_path() + 'TINYIMG',
                               train=True, download=True, transform=test_transform)
        # train = datasets.CIFAR100('Data/', train=True,  download=True)
        test = MyTinyImagenet(base_path() + 'TINYIMG',
                              train=False, download=True, transform=test_transform)
        dat['train']=train#datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=test #datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(200//t_class_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'cifar100-' + str(t_class_num*t) + '-' + str(t_class_num*(t+1)-1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_class_num*t, t_class_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 200 // t_class_num
        data[t] = {}
        data[t]['name'] = 'tiny200-all'
        data[t]['ncla'] = 200
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_tiny200_22'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_tiny200_22'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(101))
    print('Task order =', ids)
    for i in range(101):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_tiny200_22'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_tiny200_22'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(100):
        print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 64, 64))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=10,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader

    transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                     (0.2770, 0.2691, 0.2821))
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transform])
    test = MyTinyImagenet(base_path() + 'TINYIMG',
                          train=False, download=True, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader
