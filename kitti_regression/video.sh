echo $1 | xargs -I DRIVE ffmpeg -r 30 -f image2 -i images/cone-driveDRIVE-%d.png -vcodec libx264 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -crf 25  -pix_fmt yuv420p cone-driveDRIVE.mp4 -y 
rm images/cone-drive$1-*.png
