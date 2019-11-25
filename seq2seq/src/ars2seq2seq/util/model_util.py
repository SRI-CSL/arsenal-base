import os
import torch
import errno


def safe_mkdirs(dirpath):
    try:
        if not(os.path.isdir(dirpath)):
            os.makedirs(dirpath, exist_ok=True)
        return dirpath
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(dirpath):
            pass
        else:
            print("Warning, mkdirs={}, err={}".format(dirpath, exc))
            pass


def save_checkpoint(name, model, optimizer, model_dir, epoch):
    fname = os.path.join(model_dir, name)
    if not(os.path.exists(model_dir)):
        safe_mkdirs(model_dir)
    if optimizer is None:
        torch.save({'state': model.state_dict(), 'epoch': epoch}, fname)
    else:
        torch.save({'state': model.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict()}, fname)


def load_checkpoint(model, model_name, optimizer, model_dir, device=None):
    fname = os.path.join(model_dir, model_name)
    if device:
        checkpoint = torch.load(fname, map_location=device)
    else:
        checkpoint = torch.load(fname)
    model.load_state_dict(checkpoint['state'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    return epoch


def checkpoint_exists(model_name, model_dir):
    model_fname = os.path.join(model_dir, model_name)
    return os.path.isfile(model_fname)