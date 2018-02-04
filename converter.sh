count=1
for file in ../subivic/*
do
	if midi2abc "$file" -o "$count".abc; then
		echo "$count file converted"
	else
		echo "$count file failed"
	fi
	(( count++ ))
done
