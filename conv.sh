

od -An -v -tu1 -j16 -w784 train-images-idx3-ubyte | sed 's/^ *//' | tr -s ' ' >train-images.txt
od -An -v -tu1 -j8 -w1 train-labels-idx1-ubyte | tr -d ' ' >train-labels.txt
od -An -v -tu1 -j16 -w784 t10k-images-idx3-ubyte | sed 's/^ *//' | tr -s ' ' >t10k-images.txt
od -An -v -tu1 -j8 -w1 t10k-labels-idx1-ubyte | tr -d ' ' >t10k-labels.txt


file_join(){
  image=$1
  label=$2
ruby << Eof
lst=Array.new
open("$label").each do |f|
  lst.push(f.chomp)
end
open("$image").each do |f|
  print lst.shift+" "+f.chomp+"\n"
end
Eof
}

file_join train-images.txt train-labels.txt > train.txt
file_join t10k-images.txt t10k-labels.txt > t10k.txt
