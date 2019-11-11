folder=$1
python test.py  --save /home/ubuntu/save_500_new_drop_skinny_seq

cd plots
mkdir $folder
rm *.gv
mv *.eps $folder/

cd $folder

gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dEPSCrop -sOutputFile=$1.pdf *.eps

rm *.eps

cd ..
cd ..

