@echo off
echo =========================================
echo Activating Virtual Environment...
echo =========================================
call .venv\Scripts\activate

echo =========================================
echo Mixed Training: Original + Wild Data
echo ALL modules unfrozen (mask + cls + source)
echo =========================================

python run/train-AniXplore.py ^
    --model AniXplore ^
    --data_path "run/mixed_train.json" ^
    --test_data_path "run/mixed_test.json" ^
    --raw_img_data_root "E:/!project dataset/real_images" ^
    --edited_img_data_root "E:/!project dataset/fake_images" ^
	--checkpoint_path CKPTS/AniXplore/checkpoint-29.pth ^
    --batch_size 4 ^
    --test_batch_size 8 ^
    --accum_iter 1 ^
    --num_workers 4 ^
    --lr 1e-5 ^
    --epochs 10 ^
    --warmup_epochs 1 ^
    --output_dir ./my_mixed_model ^
    --if_resizing ^
    --if_test_PixelF1 ^
    --if_test_ImageF1 ^
    --if_test_ImageAccuracy ^
    --use_heavy_aug

echo =========================================
echo Task Finished!
echo =========================================
pause
