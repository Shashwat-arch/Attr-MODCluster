# Set the output format (choose one)
set terminal pngcairo enhanced font "Arial,18" size 800,600
set output 'loss_vs_epoch_cora.png'

# Alternatively, for interactive display (X11, wxt, qt, etc.)
# set terminal wxt enhanced persist

set xlabel "Epoch" font "Arial,18"
set ylabel "Loss" font "Arial,18"

# Set grid and styling
set grid
set key top right

set border linewidth 2

# Set axis ranges (optional - remove if you want auto-scaling)
#set xrange [1:150]
#set yrange [4:5]

# Plot the data
plot 'loss_epoch.txt' using 1:2 with linespoints \
     linecolor rgb "#30308a" \
     linewidth 2 \
     pointtype 7 \
     pointsize 0.8 \
     title "Training Loss"
     
