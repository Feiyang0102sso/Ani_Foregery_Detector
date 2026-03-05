@echo off
echo =========================================
echo Activating Virtual Environment...
echo =========================================
call .venv\Scripts\activate

echo =========================================
echo Phase 2: Train Source Head Only
echo Everything frozen except source_head + auto_weight
echo =========================================

python run/train-AniXplore.py ^
    --model AniXplore ^
    --data_path "run/mixed_train.json" ^
    --test_data_path "run/mixed_test.json" ^
    --raw_img_data_root "E:/!project dataset/real_images" ^
    --edited_img_data_root "E:/!project dataset/fake_images" ^
    --checkpoint_path my_mixed_model/checkpoint-9.pth ^
    --batch_size 8 ^
    --test_batch_size 8 ^
    --accum_iter 1 ^
    --num_workers 4 ^
    --lr 1e-4 ^
    --epochs 15 ^
    --warmup_epochs 1 ^
    --output_dir ./my_source_model ^
    --if_resizing ^
    --if_test_PixelF1 ^
    --if_test_ImageF1 ^
    --if_test_ImageAccuracy ^
    --use_heavy_aug ^
    --train_source_only

echo =========================================
echo Task Finished!
echo =========================================
pause

REM test_datasets_anime_train.json
