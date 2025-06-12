for f in `find . -name "*.png"`;
do
  magick convert -flatten $f $f
done