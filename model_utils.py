


def get_model(model_name, **kwargs):
    if model_name == 'VQ_VAE':
        from vqvae import VQ_VAE
        return VQ_VAE(**kwargs)
    else:
        raise ValueError(f"Model {model_name} not recognized.")