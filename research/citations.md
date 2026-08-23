# Research citations

This file records the primary literature that informed 01's architecture.
No single paper was copied. Components were selected and compared empirically.

## Global workspace and operational markers

- Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
- Dehaene, S., Kerszberg, M., & Changeux, J.-P. (1998). A neuronal model of a global workspace in effortful cognitive tasks. *PNAS*, 95(24), 14529–14534.
- Dehaene, S., & Naccache, L. (2001). Towards a cognitive neuroscience of consciousness: basic evidence and a workspace framework. *Cognition*, 79(1–2), 1–37.
- Dehaene, S., Changeux, J.-P., Naccache, L., Sackur, J., & Sergent, C. (2006). Conscious, preconscious, and subliminal processing: a testable taxonomy. *Trends in Cognitive Sciences*, 10(5), 204–211.
- Dehaene, S., & Changeux, J.-P. (2011). Experimental and theoretical approaches to conscious processing. *Neuron*, 70(2), 200–227. https://doi.org/10.1016/j.neuron.2011.03.018

Workspace ignition in 01 is an **operational analog** of GNW winner-take-all broadcasting. It is not treated as proof of phenomenal consciousness.

## Predictive processing and world models

- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*, 2(1), 79–87.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.
- Ha, D., & Schmidhuber, J. (2018). World Models. arXiv:1803.10122.
- Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Dream to Control: Learning Behaviors by Latent Imagination. *ICLR*.

## Intrinsic motivation and curiosity

- Schmidhuber, J. (1991). Curious model-building control systems. *IEEE IJCNN*.
- Oudeyer, P.-Y., Kaplan, F., & Hafner, V. V. (2007). Intrinsic motivation systems for autonomous mental development. *IEEE TAMD*, 1(2).
- Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-driven exploration by self-supervised prediction. *ICML*.
- Burda, Y., Edwards, H., Pathak, D., Storkey, A., Darrell, T., & Efros, A. A. (2019). Large-scale study of curiosity-driven learning. *ICLR*.

01 uses learning-progress curiosity and down-weights chronic novelty without progress.

## Continual learning

- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521–3526. (EWC-style importance)
- Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). Experience replay for continual learning. *NeurIPS*.

## Recurrent and continuous-time substrates

- Cho, K., et al. (2014). Learning phrase representations using RNN encoder–decoder. *EMNLP*.
- Jaeger, H. (2001). The “echo state” approach to analysing and training RNNs. GMD Report 148.
- Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021/2022). Closed-form continuous-time neural networks. *Nature Machine Intelligence*.

GRU is the default core. CfC and reservoir cells are implemented for `scripts/compare_substrates.py`. Transformers are not the language or central cognition engine. Softmax competition is used only inside the limited-capacity workspace.

## Object-centric perception and permanence

- Piaget, J. (1954). *The Construction of Reality in the Child*.
- Locatello, F., et al. (2020). Object-centric learning with slot attention. *NeurIPS*.
- Identity tracking in 01 uses Hungarian assignment on feature/location cost (Kuhn, 1955).

## Causal learning and intervention

- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press.
- Gopnik, A., et al. (2004). A theory of causal learning in children. *Psychological Review*, 111(1), 3–32.

## Self-modeling and developmental robotics

- Rochat, P. (2003). Five levels of self-awareness as they unfold early in life. *Consciousness and Cognition*, 12(4), 717–731.
- Asada, M., et al. (2009). Cognitive developmental robotics: a survey. *IEEE TAMD*, 1(1).
- Hoffmann, M., et al. (2010). Body schema and body image in robotics. *IEEE TAMD*.
- Cangelosi, A., & Schlesinger, M. (2015). *Developmental Robotics*. MIT Press.

## Grounded language

- Harnad, S. (1990). The symbol grounding problem. *Physica D*, 42, 335–346.
- Steels, L. (2015). *The Talking Heads Experiment*. Language Science Press.
- Tomasello, M. (2003). *Constructing a Language*. Harvard University Press.
- Hermann, K. M., et al. (2017). Grounded language learning in a simulated world with multiple agents. arXiv:1706.06551.

Language in 01 is a grounded I/O modality. Production requires a communicative intent and lexicon bindings. There is no next-token language model.

## Vector-symbolic and associative memory

- Plate, T. A. (1995). Holographic reduced representations. *IEEE TNN*.
- Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
- Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS*.

## Metacognition and theory of mind

- Flavell, J. H. (1979). Metacognition and cognitive monitoring. *American Psychologist*, 34(10), 906–911.
- Premack, D., & Woodruff, G. (1978). Does the chimpanzee have a theory of mind? *Behavioral and Brain Sciences*, 1(4), 515–526.
- Wimmer, H., & Perner, J. (1983). Beliefs about beliefs. *Cognition*, 13(1), 103–128.

False-belief tests in 01 are behavioral. Textbook solutions are not provided to the organism.

## Consciousness methodology (limits)

- Seth, A. K., & Bayne, T. (2022). Theories of consciousness. *Nature Reviews Neuroscience*, 23, 439–452.
- Birch, J., Schnell, A. K., & Clayton, N. S. (2020). Dimensions of animal consciousness. *Trends in Cognitive Sciences*, 24(10), 789–801.

01 maintains an evidence dossier of operational markers. It does **not** claim to have proven phenomenal consciousness.
