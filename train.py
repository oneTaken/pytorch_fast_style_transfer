import os
from model import TransformerNet
from features import Vgg16, Resnet101
from torchvision import transforms as t
from utils import *
import time


print('the current process is {}.'.format(os.getpid()))
exp_num = 3
iteration_total = 1e4
gpus = [2]
use_cuda = torch.cuda.is_available() and len(gpus)
if use_cuda:
    torch.cuda.set_device(gpus[0])

content_img_name = './images/amber.jpg'
style_img_name = './images/photo.jpg'
content_transform = t.Compose([
    t.Scale(256),
    t.CenterCrop(224),
    t.ToTensor(),
    t.Lambda(lambda x: x.mul(255))
])
style_transform = t.Compose([
    t.ToTensor(),
    t.Lambda(lambda x: x.mul(255))
])


def train(content_img_name=None, style_img_name=None, features=None):
    transformer = TransformerNet()
    # features = Vgg16()

    lr = 0.001
    weight_content = 1e5
    weight_style = 1e10
    optimizer = torch.optim.Adam(transformer.parameters(), lr)
    mse_loss = torch.nn.MSELoss()

    style = load_image(style_img_name)
    style = style_transform(style)
    style = style.unsqueeze(0)
    style_v = Variable(style)
    style_v = normalize_batch(style_v)
    features_style = features(style_v)
    gram_style = [gram_matrix(y) for y in features_style]

    transformer.train()
    x = load_image(content_img_name)
    x = content_transform(x)
    x = x.unsqueeze(0)
    x = Variable(x)

    if use_cuda:
        transformer.cuda()
        features.cuda()
        x = x.cuda()
        gram_style = [gram.cuda() for gram in gram_style]

    # training
    count = 0
    log_name = './logs/log_exp_{}.txt'.format(exp_num)
    log = []
    while count < iteration_total:
        optimizer.zero_grad()

        y = transformer(x)

        y = normalize_batch(y)
        x = normalize_batch(x)

        features_y = features(y)
        features_x = features(x)

        loss_content = mse_loss(features_y[1], features_x[1])

        loss_style = 0.
        for ft_y, gm_s in zip(features_y, gram_style):
            gm_y = gram_matrix(ft_y)
            loss_style = loss_style + mse_loss(gm_y, gm_s)

        total_loss = weight_content * loss_content + weight_style * loss_style
        total_loss.backward()
        optimizer.step()

        # log show
        count += 1
        msg = '{}\titeration: {}\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}\n'.format(
            time.ctime(), count, loss_content.data[0], loss_style.data[0], total_loss.data[0])
        log.append(msg)
        if count % 50 == 0:
            print(''.join(log))
            with open(log_name, 'a') as f:
                f.writelines(''.join(log))
                log.clear()

    # save model
    transformer.eval()
    if use_cuda:
        transformer.cpu()
    save_model_name = './models/model_exp_{}.pt'.format(exp_num)
    torch.save(transformer.state_dict(), save_model_name)


def stylize(imgname):
    transformer_name = './models/model_exp_{}.pt'.format(exp_num)
    content_image = load_image(imgname)
    content_image = style_transform(content_image)
    content_image = content_image.unsqueeze(0)
    content_image = Variable(content_image)

    transformer = TransformerNet()
    model_dict = torch.load(transformer_name)
    transformer.load_state_dict(model_dict)

    if use_cuda:
        transformer.cuda()
        content_image = content_image.cuda()

    o = transformer(content_image)
    y = o.data.cpu()[0]
    name, backend = os.path.splitext(os.path.basename(imgname))
    save_style_name = os.path.join(os.path.dirname(imgname), '{}_stylized_{}{}'.format(name, exp_num, backend))
    save_image(save_style_name, y)


if __name__ == "__main__":
    # train phase
    features = Vgg16()
    # features = Resnet101()
    train(content_img_name, style_img_name, features)
    # predict phase
    imgname = './images/amber.jpg'
    stylize(imgname)