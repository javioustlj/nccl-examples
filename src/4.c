// blocking communicators
CUDACHECK(cudaSetDevice(localRank));
for (int i = 0; i < commNum; ++i) {
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  NCCLCHECK(ncclCommInitRank(&blockingComms[i], nRanks, id, myRank));
}

// non-blocking communicators
CUDACHECK(cudaSetDevice(localRank));
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
config.blocking = 0;
for (int i = 0; i < commNum; ++i) {
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  NCCLCHECK(ncclCommInitRankConfig(&nonblockingComms[i], nRanks, id, myRank, &config));
  do {
    NCCLCHECK(ncclCommGetAsyncError(nonblockingComms[i], &state));
  } while(state == ncclInProgress && checkTimeout() != true);
}
