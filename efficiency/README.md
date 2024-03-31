Implementation efficiency evaluation requirements:
    pip install Crypten in editable mode: https://github.com/facebookresearch/CrypTen
    
To profile the 2PC bert, utilize the profile_bert.py script. Set up two machines and adjust the master address accordingly.

On the first machine, execute python profile_bert.py 0, and on the second machine, run python profile_bert.py 1.

You can experiment with different operations by modifying the values of "hidden_act", "softmax_act" and "norm" in the config class within the file.

- For activation, we support ["relu", "quad", "secformer_gelu", "puma_gelu", "crypten_gelu"]
- For softmax, we support ["softmax", "secformer_softmax", "puma_softmax", "softmax_2QUAD, "softmax_2RELU"]
- For LayerNorm, we support ["secformer_norm", "puma_norm", "crypten_norm"]
