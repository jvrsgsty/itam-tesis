data <- read.table('~/Documents/ITAM/Tesis/Batches/BatchT1_JSB/data/encoded_DBs/ggm_100A_500_200E_800G.dat', header = FALSE)
data_train_matrix <- as.matrix(data)
som_grid <- somgrid(xdim = 20, ydim=20, topo="hexagonal")

som_grid <- somgrid(xdim = 5, ydim=1, topo="rectangular")

som <- som(data_train_matrix, 
           grid=som_grid, 
           rlen=500, 
           radius=c(4.0, 0.3262874458), 
           alpha=c(0.999, 0.0065639126), 
           mode="batch",
           dist.fcts="euclidean")