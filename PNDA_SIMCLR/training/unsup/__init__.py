def setup(mode, P):
    fname = f'{P.dataset}_{P.model}_unsup_{mode}_{P.lr_init}'

    if mode == 'simclr':
        from .simclr import train
    elif mode in ['simclr_pda', 'simclr_nda', 'simclr_pnda']:
        from .simclr_pnda import train
    else:
        raise NotImplementedError()
    if P.suffix is not None:
        fname += f'_{P.suffix}'

    return train, fname

