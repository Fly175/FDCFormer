"""
CutBlur
Copyright 2020-present NAVER corp.
MIT license
"""
import numpy as np
import torch
import torch.nn.functional as F

def apply_augment(
    im1, im2,
    augs, probs, alphas,
    aux_prob=None, aux_alpha=None,
    mix_p=None
):
    idx = np.random.choice(len(augs), p=mix_p)
    aug = augs[idx]
    prob = float(probs[idx])
    alpha = float(alphas[idx])
    mask = None

    if aug == "none":
        im1_aug, im2_aug = im1.clone(), im2.clone()
    elif aug == "blend":
        im1_aug, im2_aug = blend(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha
        )
    elif aug == "mixup":
        im1_aug, im2_aug, = mixup(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha,
        )
    elif aug == "cutout":
        im1_aug, im2_aug, mask, _ = cutout(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha
        )
    elif aug == "cutmix":
        im1_aug, im2_aug = cutmix(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha,
        )
    elif aug == "cutmixup":
        im1_aug, im2_aug = cutmixup(
            im1.clone(), im2.clone(),
            mixup_prob=aux_prob, mixup_alpha=aux_alpha,
            cutmix_prob=prob, cutmix_alpha=alpha,
        )
    elif aug == "cutblur":
        im1_aug, im2_aug = cutblur(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha
        )
    elif aug == "rgb":
        im1_aug, im2_aug = rgb(
            im1.clone(), im2.clone(),
            prob=prob
        )
    else:
        raise ValueError("{} is not invalid.".format(aug))

    return im1_aug, im2_aug, mask, aug


def blend(im1, im2, prob=1.0, alpha=0.6):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    c = torch.empty((im2.size(0), 3, 1, 1), device=im2.device).uniform_(0, 255)
    rim2 = c.repeat((1, 1, im2.size(2), im2.size(3)))
    rim1 = c.repeat((1, 1, im1.size(2), im1.size(3)))

    v = np.random.uniform(alpha, 1)
    im1 = v * im1 + (1-v) * rim1
    im2 = v * im2 + (1-v) * rim2

    return im1, im2


def mixup(im1, im2, prob=1.0, alpha=1.2):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    v = np.random.beta(alpha, alpha)
    r_index = torch.randperm(im1.size(0)).to(im2.device)

    im1 = v * im1 + (1-v) * im1[r_index, :]
    im2 = v * im2 + (1-v) * im2[r_index, :]
    return im1, im2


def _cutmix(im2, prob=1.0, alpha=1.0):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return None

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)

    fcy = np.random.randint(0, h-ch+1)
    fcx = np.random.randint(0, w-cw+1)
    tcy, tcx = fcy, fcx
    rindex = torch.randperm(im2.size(0)).to(im2.device)

    return {
        "rindex": rindex, "ch": ch, "cw": cw,
        "tcy": tcy, "tcx": tcx, "fcy": fcy, "fcx": fcx,
    }


def cutmix(im1, im2, prob=1.0, alpha=1.0):
    c = _cutmix(im2, prob, alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2[rindex, :, fcy:fcy+ch, fcx:fcx+cw]
    im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[rindex, :, hfcy:hfcy+hch, hfcx:hfcx+hcw]

    return im1, im2


def cutmixup(
    im1, im2,
    mixup_prob=1.0, mixup_alpha=1.0,
    cutmix_prob=1.0, cutmix_alpha=1.0
):
    c = _cutmix(im2, cutmix_prob, cutmix_alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    v = np.random.beta(mixup_alpha, mixup_alpha)
    if mixup_alpha <= 0 or np.random.rand(1) >= mixup_prob:
        im2_aug = im2[rindex, :]
        im1_aug = im1[rindex, :]

    else:
        im2_aug = v * im2 + (1-v) * im2[rindex, :]
        im1_aug = v * im1 + (1-v) * im1[rindex, :]

    # apply mixup to inside or outside
    if np.random.random() > 0.5:
        im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2_aug[..., fcy:fcy+ch, fcx:fcx+cw]
        im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1_aug[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
    else:
        im2_aug[..., tcy:tcy+ch, tcx:tcx+cw] = im2[..., fcy:fcy+ch, fcx:fcx+cw]
        im1_aug[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
        im2, im1 = im2_aug, im1_aug

    return im1, im2


def cutblur(im1, im2, prob=1.0, alpha=1.0):
    if im1.size() != im2.size():
        raise ValueError("im1 and im2 have to be the same resolution.")

    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)
    cy = np.random.randint(0, h-ch+1)
    cx = np.random.randint(0, w-cw+1)

    # apply CutBlur to inside or outside
    if np.random.random() > 0.5:
        im2[..., cy:cy+ch, cx:cx+cw] = im1[..., cy:cy+ch, cx:cx+cw]
    else:
        im2_aug = im1.clone()
        im2_aug[..., cy:cy+ch, cx:cx+cw] = im2[..., cy:cy+ch, cx:cx+cw]
        im2 = im2_aug

    return im1, im2


def cutout(im1, im2, prob=1.0, alpha=0.1):
    scale = im1.size(2) // im2.size(2)
    fsize = (im2.size(0), 1)+im2.size()[2:]

    if alpha <= 0 or np.random.rand(1) >= prob:
        fim2 = np.ones(fsize)
        fim2 = torch.tensor(fim2, dtype=torch.float, device=im2.device)
        fim1 = F.interpolate(fim2, scale_factor=scale, mode="nearest")
        return im1, im2, fim1, fim2

    fim2 = np.random.choice([0.0, 1.0], size=fsize, p=[alpha, 1-alpha])
    fim2 = torch.tensor(fim2, dtype=torch.float, device=im2.device)
    fim1 = F.interpolate(fim2, scale_factor=scale, mode="nearest")

    im2 *= fim2

    return im1, im2, fim1, fim2


def rgb(im1, im2, prob=1.0):
    if np.random.rand(1) >= prob:
        return im1, im2

    perm = np.random.permutation(3)
    im1 = im1[:, perm]
    im2 = im2[:, perm]

    return im1, im2



class Solver():
    def __init__(self, module, opt):
        self.opt = opt

        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = module.Net(opt).to(self.dev)
        print("# params:", sum(map(lambda x: x.numel(), self.net.parameters())))

        if opt.pretrain:
            self.load(opt.pretrain)

        self.loss_fn = nn.L1Loss()
        self.optim = torch.optim.Adam(
            self.net.parameters(), opt.lr,
            betas=(0.9, 0.999), eps=1e-8
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optim, [1000*int(d) for d in opt.decay.split("-")],
            gamma=opt.gamma,
        )

        if not opt.test_only:
            self.train_loader = generate_loader("train", opt)
        self.test_loader = generate_loader("test", opt)

        self.t1, self.t2 = None, None
        self.best_psnr, self.best_step = 0, 0

    def fit(self):
        opt = self.opt

        self.t1 = time.time()
        for step in range(opt.max_steps):
            try:
                inputs = next(iters)
            except (UnboundLocalError, StopIteration):
                iters = iter(self.train_loader)
                inputs = next(iters)

            HR = inputs[0].to(self.dev)
            LR = inputs[1].to(self.dev)


            # match the resolution of (LR, HR) due to CutBlur
            if HR.size() != LR.size():
                scale = HR.size(2) // LR.size(2)
                LR = F.interpolate(LR, scale_factor=scale, mode="nearest")

            HR, LR, mask, aug = augments.apply_augment(
                HR, LR,
                opt.augs, opt.prob, opt.alpha,
                opt.aux_alpha, opt.aux_alpha, opt.mix_p
            )

            SR = self.net(LR) #
            if aug == "cutout":
                SR, HR = SR*mask, HR*mask

            loss = self.loss_fn(SR, HR)
            self.optim.zero_grad()
            loss.backward()

            if opt.gclip > 0:
                torch.nn.utils.clip_grad_value_(self.net.parameters(), opt.gclip)

            self.optim.step()
            self.scheduler.step()

            if (step+1) % opt.eval_steps == 0:
                self.summary_and_save(step)