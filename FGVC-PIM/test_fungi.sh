python test_fungi.py /home/data1/lkd/CVPR_FUNGI/Fungi_data/Fungidata/DF21 \
            --model swin-t \
            --img-size 384 \
            --num-classes 1604 \
            --checkpoint result/fungi_swin_large/backup/last.pt \
            --output_path ./output/swint_large_fungi1.csv \
            --scoreoutput_path ./output/swint_large_fungi_score1.csv \
            --batch-size 16 \
            --crop-pct 1.0 \
            --mode 1
