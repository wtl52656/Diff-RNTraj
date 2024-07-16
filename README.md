# Diff-RocT
The code of Diff-RNTraj
# Training:
  Running the following command to train Diff-RNTraj.
  
  **Porto dataset:**     python multi_main.py --dataset Porto --diff_T 500 --pre_trained_dim 64 --rdcl 10
  
  **Chengdu dataset:**     python multi_main.py --dataset Chengdu --diff_T 500 --pre_trained_dim 64 --rdcl 10

# Generate data:
  After training the Diff-RNTraj, run the following command to generate the road network-constrained trajectory.
  
  **Porto dataset:**     python generate_data.py --dataset Porto --diff_T 500 --pre_trained_dim 64 --rdcl 10

  **Chengdu dataset:**     python generate_data.py --dataset Chengdu --diff_T 500 --pre_trained_dim 64 --rdcl 10


# Generated Sample:
   Our generated trajectories on the Chengdu and Porto datasets in the path of **./generate_data**.
