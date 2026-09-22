   59  git clone https://github.com/sbsiewertcsu/numeric-parallel-starter-code.git
   62  cd numeric-parallel-starter-code/
   64  cd hello_cluster/
    7  mpirun -np 8 --map-by ppr:8:node --hostfile c1_hosts.sam ./greetings
   12  make clean
   13  make
   15  mpirun -np 4 --map-by ppr:4:node --hostfile c1_hosts.sam ./greetings
   18  mpirun -np 8 --map-by ppr:4:node --hostfile c1_hosts.sam ./greetings
   19  mpirun -np 16 --map-by ppr:4:node --hostfile c1_hosts.sam ./greetings
   68  mpiexec ./greetings
   69  mpirun -np 8 --map-by ppr:8:node --host rpi1:4 --host rpi3:4 ./greetings
   70  mpirun -np  --map-by ppr:8:node --host rpi1:4 --host rpi3:4 ./greetings
   71  mpirun --mca btl_tcp_if_include eth0 -np 8 --host rpi1:4 --host rpi3:4 ./greetings
   85  mpirun -np 8 --map-by ppr:8:node --hostfile c1_hosts.sam ./greetings
   87  mpirun -np 8 --map-by ppr:8:node --hostfile c1_hosts.sam ./greetings
  103  mpirun -np 16 --map-by ppr:4:node --hostfile rpi_hosts.sam ./greetingscpp
  104  mpirun -np 16 --map-by ppr:4:node --hostfile rpi_hosts.sam ./greetings
  111  mpiexec ./greetings
  112  mpirun -np 16 --map-by ppr:4:node --hostfile rpi_hosts.sam ./greetings
  113  mpirun --map-by ppr:4:node --hostfile rpi_hosts.sam ./greetings
  115  mpirun --map-by ppr:4:node --hostfile rpi_hosts.sam ./piseriessimp 1000000
  118  mpirun --map-by ppr:4:node --hostfile rpi_hosts.sam ./greetingscpp
  119  mpirun --map-by ppr:4:node --hostfile rpi_hosts.sam ./compare
  121  mpirun --map-by ppr:4:node --hostfile rpi_hosts.sam ./hybridcompare
  123  mpirun --map-by ppr:4:node --hostfile rpi_hosts.sam ./mpi_array
  125  mpirun --map-by ppr:4:node --hostfile rpi_hosts.sam ./piseriesreduce
  126  mpirun --map-by ppr:4:node --hostfile rpi_hosts.sam ./piseriesreduce 1000000
  128  mpirun --map-by ppr:4:node --hostfile rpi_hosts.sam ./rankmul
  129  mpirun --map-by ppr:4:node --hostfile rpi_hosts.sam ./rankmulallreduce
  131  mpirun --map-by ppr:4:node --hostfile rpi_hosts.sam ./ranksumbutterfly
  149  mpirun --map-by ppr:4:node --hostfile rpi_hosts.sam ./ranksumbutterfly
  153  cd MPI_Examples/
  159  make clean
  165  make
  171  cp ../hello_cluster/rpi_hosts.sam .
  172  mpirun --map-by ppr:4:node --hostfile rpi_hosts.sam ./mpi_trap1
  173  mpirun --map-by ppr:4:node --hostfile rpi_hosts.sam ./mpi_trap3
  177  mpirun --map-by ppr:4:node --hostfile rpi_hosts.sam ./mpi_trap3
  222  mpirun --map-by ppr:4:node --hostfile c1_hosts.first4 ./greetings
  227  mpirun --map-by ppr:4:node --hostfile rpi_hosts.first4 ./greetings
  228  mpirun ---np 16 map-by ppr:4:node --hostfile rpi_hosts.first4 ./greetings
  229  mpirun --np 16 map-by ppr:4:node --hostfile rpi_hosts.first4 ./greetings
  233  mpirun --np 16 map-by ppr:4:node --hostfile rpi_hosts.sam ./greetings
  266  mpirun --map-by ppr:4:node --hostfile rpi_hosts.sam ./piseriessimp 1000000
  268  mpirun --map-by ppr:4:node --hostfile rpi_hosts.sam ./greetings
