@echo off
echo =========================================
echo Activating Virtual Environment...
echo =========================================
call .venv\Scripts\activate

echo =========================================
echo Testing AniXplore: Full Metrics + Confusion Matrix
echo =========================================

python run/test-AniXplore.py ^
    --model AniXplore ^
    --checkpoint_path uploaded_model/checkpoint-8_260306_2018.pth ^
    --raw_img_data_root "E:/!project dataset/real_images" ^
    --edited_img_data_root "E:/!project dataset/fake_images" ^
    --test_data_json "run/test_datasets_anime_test.json" ^
    --test_batch_size 8 ^
    --if_resizing ^
    --image_size 512 ^
    --output_dir ./test_results_confusion ^
    --log_dir ./test_results_confusion ^
    --if_test_ImageF1 ^
    --if_test_PixelF1 ^
    --if_test_ImageAccuracy ^
    --if_test_PixelAccuracy ^
    --if_test_PixelIOU ^
    --if_test_ImageAUC ^
    --if_test_PixelAUC ^
    --if_test_ImageConfusionMatrix ^
    --if_test_PixelConfusionMatrix ^
    --if_test_SourceAccuracy ^
    --if_test_SourcePrecision ^
    --if_test_SourceRecall ^
    --if_test_SourceF1 ^
    --if_test_SourceConfusionMatrix

echo =========================================
echo Task Finished!
echo =========================================
pause
