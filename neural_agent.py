"""
neural_agent.py
===============
Réseau de neurones simple utilisé comme GÉNOME de l'algorithme génétique.

Architecture : 12 → 24 → 16 → 4  (fully-connected, activation tanh)

Le GÉNOME est le vecteur aplati de tous les poids et biais du réseau.
L'AG évoluera ce vecteur de paramètres sans rétropropagation.
"""

import numpy as np


class NeuralAgent:
    """
    Réseau de neurones feed-forward dont les poids constituent le génome.

    Architecture fixe : input(12) → hidden1(24) → hidden2(16) → output(4)
    Activation : tanh (couches cachées) + argmax (sortie)
    """

    LAYER_SIZES = [12, 24, 16, 4]  # architecture du réseau

    def __init__(self, genome=None):
        """
        Paramètres
        ----------
        genome : np.ndarray ou None
            Vecteur 1D de poids. Si None, initialisation aléatoire.
        """
        self.genome_size = self._compute_genome_size()

        if genome is not None:
            self.genome = np.array(genome, dtype=np.float32)
        else:
            # Initialisation He (bien adaptée à tanh)
            self.genome = np.random.randn(self.genome_size).astype(np.float32) * 0.5

        self._unpack_weights()

    # ──────────────────────────────────────────
    @classmethod
    def _compute_genome_size(cls):
        """Calcule la taille totale du génome (poids + biais de chaque couche)."""
        total = 0
        sizes = cls.LAYER_SIZES
        for i in range(len(sizes) - 1):
            total += sizes[i] * sizes[i + 1]  # poids
            total += sizes[i + 1]              # biais
        return total

    def _unpack_weights(self):
        """Découpe le génome 1D en matrices W et vecteurs b pour chaque couche."""
        self.weights = []
        self.biases  = []
        sizes  = self.LAYER_SIZES
        cursor = 0
        for i in range(len(sizes) - 1):
            n_in  = sizes[i]
            n_out = sizes[i + 1]
            w_size = n_in * n_out
            W = self.genome[cursor: cursor + w_size].reshape(n_in, n_out)
            cursor += w_size
            b = self.genome[cursor: cursor + n_out]
            cursor += n_out
            self.weights.append(W)
            self.biases.append(b)

    # ──────────────────────────────────────────
    def forward(self, x):
        """
        Propagation avant.

        x      : np.ndarray shape (12,)
        return : int (action 0-3)
        """
        h = x.astype(np.float32)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W + b
            if i < len(self.weights) - 1:
                h = np.tanh(h)          # activation cachée
        return int(np.argmax(h))        # action = argmax de la sortie

    def act(self, obs):
        """Interface compatible avec evaluate_agent()."""
        return self.forward(obs)

    # ──────────────────────────────────────────
    def copy(self):
        """Retourne une copie indépendante de l'agent."""
        return NeuralAgent(genome=self.genome.copy())

    def set_genome(self, genome):
        """Met à jour le génome et reconstruit les poids."""
        self.genome = np.array(genome, dtype=np.float32)
        self._unpack_weights()

    # ──────────────────────────────────────────
    def __repr__(self):
        return (f"NeuralAgent(arch={self.LAYER_SIZES}, "
                f"genome_size={self.genome_size})")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = NeuralAgent()
    print(agent)
    print(f"Taille du génome : {agent.genome_size} paramètres")

    # Test forward pass
    obs = np.random.rand(12).astype(np.float32)
    action = agent.act(obs)
    print(f"Action produite  : {action}  (0=Haut, 1=Bas, 2=Gauche, 3=Droite)")
    print("✅ NeuralAgent OK !")
