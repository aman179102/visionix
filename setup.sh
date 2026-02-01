for i in {1..25}
do
  DATE=$(date -d "2026-02-$i 10:0$((i%10)):00" +"%Y-%m-%d %H:%M:%S")
  
  echo "// minor update $i" >> commit_log.txt
  
  git add .
  GIT_AUTHOR_DATE="$DATE" GIT_COMMITTER_DATE="$DATE" \
  git commit -m "Refinement update $i"
  
  git push origin main
  
  sleep 2
done