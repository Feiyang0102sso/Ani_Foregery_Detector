@echo off
echo =========================================
echo Activating Virtual Environment...
echo =========================================
call .venv\Scripts\activate

echo =========================================
echo Starting AniXplore Training from Scratch!
echo =========================================

python run/train-AniXplore.py ^
    --model AniXplore ^
    --data_path "run/test_datasets_anime_train.json" ^
    --test_data_path "run/test_datasets_anime_test.json" ^
    --raw_img_data_root "E:/!project dataset/real_images" ^
    --edited_img_data_root "E:/!project dataset/fake_images" ^
	--checkpoint_path CKPTS/AniXplore/checkpoint-29.pth ^
    --batch_size 2 ^
    --accum_iter 4 ^
    --num_workers 4 ^
    --lr 1e-5 ^
    --epochs 10 ^
    --warmup_epochs 1 ^
    --output_dir ./my_new_shearlet_model ^
    --if_resizing ^
    --if_test_PixelF1 ^
    --if_test_ImageF1 ^
    --if_test_ImageAccuracy


echo =========================================
echo Task Finished!
echo =========================================
pause

REM --checkpoint_path CKPTS/AniXplore/checkpoint-29.pth ^
REM --freeze_backbone
REM test_datasets_anime_train.json
REM test_datasets_anime_test.json
REM test_datasets_anime_train.json