import torch
from utils.temperature_scaling import _ECELoss
from utils.utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ece_criterion = _ECELoss().to(device)


def error_k(output, target, ks=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(ks)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in ks:
        correct_k = correct[:k].view(-1).float().sum(0)
        results.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return results


def test_classifier(P, model, loader, steps, marginal=False, logger=None):
    error_top1 = AverageMeter()
    error_calibration = AverageMeter()

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    # Switch to evaluate mode
    mode = model.training
    model.eval()

    for n, (images, labels) in enumerate(loader):
        batch_size = images.size(0)

        images, labels = images.to(device), labels.to(device)

        if marginal:
            outputs = 0
            for i in range(4):
                rot_images = torch.rot90(images, i, (2, 3))
                _, outputs_aux = model(rot_images, joint=True)
                outputs += outputs_aux['joint'][:, P.n_classes * i: P.n_classes * (i + 1)] / 4.
        else:
            outputs = model(images)

        top1, = error_k(outputs.data, labels, ks=(1,))
        error_top1.update(top1.item(), batch_size)

        ece = ece_criterion(outputs, labels) * 100
        error_calibration.update(ece.item(), batch_size)

        if n % 100 == 0:
            log_('[Test %3d] [Test@1 %.3f] [ECE %.3f]' %
                 (n, error_top1.value, error_calibration.value))

    log_(' * [Error@1 %.3f] [ECE %.3f]' %
         (error_top1.average, error_calibration.average))

    if logger is not None:
        logger.scalar_summary('eval/clean_error', error_top1.average, steps)
        logger.scalar_summary('eval/ece', error_calibration.average, steps)

    model.train(mode)

    return error_top1.average
