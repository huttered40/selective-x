





initialize(){
  .. this was in accelerate::initialize(...)
  //TODO: Should we still do this?
  generate_initial_aggregate();

  kernel_propagate ex_3;
  MPI_Datatype kernel_internal_type[2] = { MPI_INT, MPI_FLOAT };
  int kernel_internal_block_len[2] = { 3,6 };
  MPI_Aint kernel_internal_disp[2] = { (char*)&ex_3.hash_id-(char*)&ex_3, (char*)&ex_3.num_scheduled_units-(char*)&ex_3 };
  PMPI_Type_create_struct(2,kernel_internal_block_len,kernel_internal_disp,kernel_internal_type,&kernel_type);
  PMPI_Type_commit(&kernel_type);

}
