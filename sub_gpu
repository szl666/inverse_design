#!/bin/bash

#SBATCH -J zhilong
#SBATCH -p gpu4
#SBATCH --cpus-per-task=10
#SBATCH--reservation=root_46
#SBATCH --gres=gpu:1 

module load cuda/11.3
#python run1.py data=catalyst expname=catalyst
#python run1.py data=catalyst_oqmd expname=catalyst_oqmd
#python -u -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 main.py --distributed --num-gpus 2 --mode train --config-yml configs/is2re/all/painn/painn_h1024_bs8x4.yml
#python main.py --mode train --config-yml configs/is2re/all/painn/painn_h1024_bs8x4.yml
#python main.py --mode train --config-yml configs/is2re/all/gemnet_oc/gemnet-oc.yml
#python main.py --mode train --config-yml configs/is2re/all/dimenet_plus_plus/dpp.yml
#python main.py --mode train --config-yml configs/is2re/all/dimenet_plus_plus/dpp_mbj.yml
#python main.py --mode train --config-yml configs/is2re/all/dimenet_plus_plus/dpp_gllbsc.yml
#python main.py --mode train --config-yml configs/is2re/all/dimenet_plus_plus/dpp_exp.yml
#python main.py --mode train --config-yml configs/is2re/all/dimenet_plus_plus/dpp_hse.yml
#python -u -m torch.distributed.launch --nproc_per_node=2 --master_port 29504 main.py --distributed --num-gpus 2 --mode train --config-yml configs/is2re/all/dimenet_plus_plus/dpp.yml
#python -u -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 main.py --distributed --num-gpus 2 --mode train --config-yml configs/is2re/all/gemnet_oc/gemnet-oc.yml
#python -u -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 main.py --distributed --num-gpus 2 --mode train --config-yml configs/is2re/all/gemnet_oc/gemnet-oc-large.yml
#python -u -m torch.distributed.launch --nproc_per_node=2 main.py --distributed --num-gpus 2 --mode train --config-yml configs/is2re/all/gemnet_dt/gemnet-dt.yml
#python main.py --mode train --config-yml configs/is2re/all/gemnet_dt/gemnet-dt.yml
#python -u -m torch.distributed.launch --nproc_per_node=2 main.py --distributed --num-gpus 2 --mode train --config-yml configs/is2re/all/scn/scn.yml
#python -u -m torch.distributed.launch --nproc_per_node=2 --master_port 29504 main.py --distributed --num-gpus 2 --mode train --config-yml configs/is2re/all/cgcnn/cgcnn.yml
#python compute_metrics.py --root_path /seu_share/home/WL-liqiang/olddata/WL-liqiang/workspace/zlsong/cdvae-main/hydra/singlerun/2022-10-25/catalyst --tasks gen
#python evaluate_guangcui.py --model_path /fs0/home/liqiang/onega_test/hydra/singlerun/2022-12-05/catalyst/ --tasks opt --start_from no
#python evaluate_guangcui.py --model_path /fs0/home/liqiang/onega_test/hydra/singlerun/2022-11-14/catalyst/ --tasks opt --start_from no
#python evaluate_guangcui.py --model_path /fs0/home/liqiang/onega_test/hydra/singlerun/2023-04-03/catalyst/ --tasks opt --start_from no
python evaluate_guangcui.py --model_path /fs0/home/liqiang/onega_test/hydra/singlerun/2023-04-11/catalyst_oqmd/ --tasks opt --start_from no
