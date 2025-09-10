# Set output format and file
set terminal pngcairo enhanced font "Arial,18" size 800,600
set output 'metrics_vs_epoch.png'

# Set labels and title
set xlabel "Epoch" font "Arial,18"
set ylabel "Metric Value" font "Arial,18"
#set title "Epoch vs ACC, NMI, ARI, and F1" font "Arial,16"

# Enable grid and border
set grid
set border linewidth 2

# Configure legend
set key top left

# Set line styles for each metric with different colors
set style line 1 lc rgb "#1f77b4" lw 2 pt 7 ps 1.5  # Blue: ACC
set style line 2 lc rgb "#ff7f0e" lw 2 pt 5 ps 1.5  # Orange: NMI
set style line 3 lc rgb "#2ca02c" lw 2 pt 9 ps 1.5  # Green: ARI
set style line 4 lc rgb "#d62728" lw 2 pt 11 ps 1.5 # Red: F1

# Plot all four metrics vs Epoch from the data file 'metrics_epoch.txt'
plot 'epoch_performance.txt' using 1:2 with lines linestyle 1 title "ACC", \
     '' using 1:3 with lines linestyle 2 title "NMI", \
     '' using 1:4 with lines linestyle 3 title "ARI", \
     '' using 1:5 with lines linestyle 4 title "F1"

