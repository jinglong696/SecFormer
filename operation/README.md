Implementation operation evaluation requirements:

    pip install Crypten in editable mode: https://github.com/facebookresearch/CrypTen

- Evaluation of GeLU: 
To profile the 2PC gelu, utilize the gelu.py script. Set up two machines and adjust the master address accordingly.
On the first machine, execute python gelu.py 0, and on the second machine, run python gelu.py 1.

- Evaluation of LayerNorm: 
To profile the 2PC layernorm, utilize the layernorm.py script. Set up two machines and adjust the master address accordingly.
On the first machine, execute python layernorm.py 0, and on the second machine, run python layernorm.py 1.

- Evaluation of Softmax: 
To profile the 2PC softmax, utilize the softmax.py script. Set up two machines and adjust the master address accordingly.
On the first machine, execute python softmax.py 0, and on the second machine, run python softmax.py 1.

- Evaluation of division: 
To profile the 2PC division, utilize the division.py script. Set up two machines and adjust the master address accordingly.
On the first machine, execute python division.py 0, and on the second machine, run python division.py 1.

- Evaluation of inverse square root: 
To profile the 2PC inverse square root, utilize the inv_sqrt.py script. Set up two machines and adjust the master address accordingly.
On the first machine, execute python inv_sqrt.py 0, and on the second machine, run python inv_sqrt.py 1.