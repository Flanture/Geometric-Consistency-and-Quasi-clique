# CUDA_VISIBLE_DEVICES=0 python test.py --snapshot=/mnt/sdb/public/data/fyc/gitspace/GeoTransformer/output/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/snapshots/500snapshot.pth.tar --benchmark=3DLoMatch 
# CUDA_VISIBLE_DEVICES=0 python test.py --snapshot=/mnt/sdb/public/data/fyc/gitspace/GeoTransformer/output/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/snapshots/500snapshot.pth.tar --benchmark=3DMatch 

# CUDA_VISIBLE_DEVICES=0 python test.py --snapshot=/mnt/sdb/public/data/fyc/gitspace/GeoTransformer/weights/snapshot-20240607.pth.tar --benchmark=3DMatch > 50003dlomatch.log


# CUDA_VISIBLE_DEVICES=1 python test.py --snapshot=../../weights/snapshot-20240607.pth.tar --benchmark=3DLoMatch
# CUDA_VISIBLE_DEVICES=1 python test.py --snapshot=../../weights/snapshot-20240607.pth.tar --benchmark=3DMatch
CUDA_VISIBLE_DEVICES=2 python test.py --snapshot=/mnt/sdb/public/data/fyc/gitspace/GeoTransformer/output/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/snapshots/2500snapshot.pth.tar --benchmark=3DLoMatch