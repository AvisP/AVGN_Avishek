import tensorflow as tf
from tensorflow_probability.python.distributions import Chi2
import matplotlib.pyplot as plt


class GAIA(tf.keras.Model):
    """a basic gaia class for tensorflow

    Extends:
        tf.keras.Model
    """

    def __init__(self, d_prop_xg=1.0, g_prop_interp=1.0, **kwargs):
        super(GAIA, self).__init__()
        self.__dict__.update(kwargs)
        self.d_prop_xg = d_prop_xg
        self.g_prop_interp = g_prop_interp
        self.enc = tf.keras.Sequential(self.enc)
        self.dec = tf.keras.Sequential(self.dec)

        inputs, outputs = self.unet_function()
        self.disc = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def discriminate(self, x):
        return self.disc(x)

    def regularization(self, x1, x2):
        return tf.reduce_mean(tf.square(x1 - x2))

    @tf.function
    def network_pass(self, x):
        z = self.encode(x)
        xg = self.decode(z)
        zi = self._interpolate_z(z)
        xi = self.decode(zi)
        d_xi = self.discriminate(xi)
        d_x = self.discriminate(x)
        d_xg = self.discriminate(xg)
        return z, xg, zi, xi, d_xi, d_x, d_xg

    @tf.function
    def compute_loss(self, x):
        # run through network
        z, xg, zi, xi, d_xi, d_x, d_xg = self.network_pass(x)

        # compute losses
        X_G_X_loss = tf.clip_by_value(self.regularization(x, xg), 0, 1)
        D_X_D_G_X_loss = tf.clip_by_value(self.regularization(d_x, d_xg), 0, 1)
        X_D_G_X_loss = tf.clip_by_value(self.regularization(x, d_xg), 0, 1)
        X_D_X_G_X_loss = tf.clip_by_value(self.regularization(xg, d_xg), 0, 1)
        X_D_G_Zi_loss = tf.clip_by_value(self.regularization(xi, d_xi), 0, 1)
        X_G_loss = (X_D_G_Zi_loss + X_D_G_X_loss) / 2.0
        X_D_X_loss = tf.clip_by_value(self.regularization(x, d_x), 0, 1)

        # add losses for generator and descriminator
        # loss of Encoder/Decoder: reconstructing x_real well and x_fake poorly
        # maximize the difference between data - dist_data and inter - disc_interp
        D_loss = (
            X_D_X_loss
            + X_D_G_X_loss
            - tf.clip_by_value(X_D_G_Zi_loss, 0, X_D_X_loss) * self.d_prop_xg
        )

        # Generator should be balancing the reproduction
        # minimize the difference between interp-disc-interp and gen - disc-gen
        G_loss = (
            X_G_X_loss
            + tf.clip_by_value(X_D_G_Zi_loss, 0, X_D_X_loss) * self.g_prop_interp
        )

        return (
            X_D_G_X_loss,
            X_D_G_Zi_loss,
            X_G_loss,
            X_D_X_loss,
            X_D_X_G_X_loss,
            X_G_X_loss,
            G_loss,
            D_loss,
        )

    @tf.function
    def compute_gradients(self, x):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            _, _, _, _, _, _, G_loss, D_loss = self.compute_loss(x)

            gen_loss = G_loss
            disc_loss = D_loss

        gen_gradients = gen_tape.gradient(
            gen_loss, self.enc.trainable_variables + self.dec.trainable_variables
        )
        disc_gradients = disc_tape.gradient(disc_loss, self.disc.trainable_variables)
        return gen_gradients, disc_gradients

    @tf.function
    def apply_gradients(self, gen_gradients, disc_gradients):
        self.gen_optimizer.apply_gradients(
            zip(
                gen_gradients,
                self.enc.trainable_variables + self.dec.trainable_variables,
            )
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.disc.trainable_variables)
        )

    @tf.function
    def train_net(self, x):
        gen_gradients, disc_gradients = self.compute_gradients(x)
        self.apply_gradients(gen_gradients, disc_gradients)

    def _interpolate_z(self, z):
        """ takes the dot product of some random tensor of batch_size,
         and the z representation of the batch as the interpolation
        """
        if self.chsq.df != z.shape[0]:
            self.chsq = Chi2(df=1 / z.shape[0])
        ip = self.chsq.sample((z.shape[0], z.shape[0]))
        ip = ip / tf.reduce_sum(ip, axis=0)
        zi = tf.transpose(tf.tensordot(tf.transpose(z), ip, axes=1))
        return zi


def sigmoid(x, shift=0.0, mult=20):
    """ squashes a value with a sigmoid
    """
    return tf.constant(1.0) / (
        tf.constant(1.0) + tf.exp(-tf.constant(1.0) * ((x + tf.constant(shift)) * mult))
    )


def plot_reconstruction(model, example_data, nex=5, zm=3):
    z, xg, zi, xi, d_xi, d_x, d_xg = model.network_pass(example_data)
    fig, axs = plt.subplots(ncols=6, nrows=nex, figsize=(zm * 6, zm * nex))
    for axi, (dat, lab) in enumerate(
        zip(
            [example_data, d_x, xg, d_xg, xi, d_xi],
            ["data", "disc data", "gen", "disc gen", "interp", "disc interp"],
        )
    ):
        for ex in range(nex):
            axs[ex, axi].matshow(
                dat.numpy()[ex].squeeze(), cmap=plt.cm.Greys, vmin=0, vmax=1
            )
            axs[ex, axi].axis("off")
        axs[0, axi].set_title(lab)

    plt.show()

