#ifndef __SELECTIVEX__MPI_H_
#define __SELECTIVEX__MPI_H_

#include "../src/interface.h"
#include "../src/intercept/comm.h"

#define MPI_Init(argc, argv)\
     selectivex_init(argc,argv)

#define MPI_Init_thread(argc, argv, required, provided)\
     selectivex_init_thread(argc,argv,required,provided)

#define MPI_Finalize()\
    selectivex_finalize()

#define MPI_Barrier(cm)\
    selectivex_barrier(cm)

#define MPI_Comm_split(comm,color,key,newcomm)\
    selectivex_comm_split(comm,color,key,newcomm)

#define MPI_Comm_dup(comm,newcomm)\
    selectivex_comm_dup(comm,newcomm)

#define MPI_Comm_free(cm)\
    selectivex_comm_free(cm)

#define MPI_Bcast(buf, nelem, t, root, cm)\
    selectivex_bcast(buf,nelem,t,root,cm)

#define MPI_Reduce(sbuf, rbuf, nelem, t, op, root, cm)\
    selectivex_reduce(sbuf,rbuf,nelem,t,op,root,cm)

#define MPI_Allreduce(sbuf, rbuf, nelem, t, op, cm)\
    selectivex_allreduce(sbuf,rbuf,nelem,t,op,cm)

#define MPI_Gather(sbuf, scount, st, rbuf, rcount, rt, root, cm)\
    selectivex_gather(sbuf,scount,st,rbuf,rcount,rt,root,cm)

#define MPI_Allgather(sbuf, scount, st, rbuf, rcount, rt, cm)\
    selectivex_allgather(sbuf,scount,st,rbuf,rcount,rt,cm)

#define MPI_Scatter(sbuf, scount, st, rbuf, rcount, rt, root, cm)\
    selectivex_scatter(sbuf,scount,st,rbuf,rcount,rt,root,cm)

#define MPI_Reduce_scatter(sbuf, rbuf, rcounts, t, op, cm)\
    selectivex_reduce_scatter(sbuf,rbuf,rcounts,t,op,cm)

#define MPI_Alltoall(sbuf, scount, st, rbuf, rcount, rt, cm)\
    selectivex_alltoall(sbuf,scount,st,rbuf,rcount,rt,cm)

#define MPI_Gatherv(sbuf, scount, st, rbuf, rcounts, rdispsls, rt, root, cm)\
    selectivex_gatherv(sbuf,scount,st,rbuf,rcounts,rdispsls,rt,root,cm)

#define MPI_Allgatherv(sbuf, scount, st, rbuf, rcounts, rdispsls, rt, cm)\
    selectivex_allgatherv(sbuf,scount,st,rbuf,rcounts,rdispsls,rt,cm)

#define MPI_Scatterv(sbuf, scounts, sdispls, st, rbuf, rcount, rt, root, cm)\
    selectivex_scatterv(sbuf,scounts,sdispls,st,rbuf,rcount,rt,root,cm)

#define MPI_Alltoallv(sbuf, scounts, sdispls, st, rbuf, rcounts, rdispsls, rt, cm)\
    selectivex_alltoallv(sbuf,scounts,sdispls,st,rbuf,rcounts,rdispsls,rt,cm)

#define MPI_Ibcast(buf, nelem, t, root, cm, req)\
    selectivex_ibcast(buf,nelem,t,root,cm,req)

#define MPI_Iallreduce(sbuf, rbuf, nelem, t, op, cm, req)\
    selectivex_iallreduce(sbuf,rbuf,nelem,t,op,cm,req)

#define MPI_Ireduce(sbuf, rbuf, nelem, t, op, root, cm, req)\
    selectivex_ireduce(sbuf,rbuf,nelem,t,op,root,cm,req)

#define MPI_Igather(sbuf, scount, st, rbuf, rcount, rt, root, cm, req)\
    selectivex_igather(sbuf,scount,st,rbuf,rcount,rt,root,cm,req)

#define MPI_Igatherv(sbuf, scount, st, rbuf, rcounts, rdispsls, rt, root, cm, req)\
    selectivex_igatherv(sbuf,scount,st,rbuf,rcounts,rdispsls,rt,root,cm,req)

#define MPI_Iallgather(sbuf, scount, st, rbuf, rcount, rt, cm, req)\
    selectivex_iallgather(sbuf,scount,st,rbuf,rcount,rt,cm,req)

#define MPI_Iallgatherv(sbuf, scount, st, rbuf, rcounts, rdispsls, rt, cm, req)\
    selectivex_iallgatherv(sbuf,scount,st,rbuf,rcounts,rdispsls,rt,cm,req)

#define MPI_Iscatter(sbuf, scount, st, rbuf, rcount, rt, root, cm, req)\
    selectivex_iscatter(sbuf,scount,st,rbuf,rcount,rt,root,cm,req)

#define MPI_Iscatterv(sbuf, scounts, sdispls, st, rbuf, rcount, rt, root, cm, req)\
    selectivex_iscatterv(sbuf,scounts,sdispls,st,rbuf,rcount,rt,root,cm,req)

#define MPI_Ireduce_scatter(sbuf, rbuf, rcounts, t, op, cm, req)\
    selectivex_ireduce_scatter(sbuf,rbuf,rcounts,t,op,cm,req)

#define MPI_Ialltoall(sbuf, scount, st, rbuf, rcount, rt, cm, req)\
    selectivex_ialltoall(sbuf,scount,st,rbuf,rcount,rt,cm,req)

#define MPI_Ialltoallv(sbuf, scounts, sdispls, st, rbuf, rcounts, rdispsls, rt, cm, req)\
    selectivex_ialltoallv(sbuf,scounts,sdispls,st,rbuf,rcounts,rdispsls,rt,cm,req)

#define MPI_Sendrecv(sbuf, scnt, st, dest, stag, rbuf, rcnt, rt, src, rtag, cm, status)\
    selectivex_sendrecv(sbuf,scnt,st,dest,stag,rbuf,rcnt,rt,src,rtag,cm,status)

#define MPI_Sendrecv_replace(sbuf, scnt, st, dest, stag, src, rtag, cm, status)\
    selectivex_sendrecv_replace(sbuf,scnt,st,dest,stag,src,rtag,cm,status)

#define MPI_Ssend(buf, nelem, t, dest, tag, cm)\
    selectivex_ssend(buf,nelem,t,dest,tag,cm)

#define MPI_Bsend(buf, nelem, t, dest, tag, cm)\
    selectivex_bsend(buf,nelem,t,dest,tag,cm)

#define MPI_Send(buf, nelem, t, dest, tag, cm)\
    selectivex_send(buf,nelem,t,dest,tag,cm)

#define MPI_Recv(buf, nelem, t, src, tag, cm, status)\
    selectivex_recv(buf,nelem,t,src,tag,cm,status)

#define MPI_Isend(buf, nelem, t, dest, tag, cm, req)\
    selectivex_isend(buf,nelem,t,dest,tag,cm,req)

#define MPI_Irecv(buf, nelem, t, src, tag, cm, req)\
    selectivex_irecv(buf, nelem, t, src, tag, cm, req)

#define MPI_Wait(req, stat)\
    selectivex_wait(req,stat)

#define MPI_Waitany(cnt, reqs, indx, stat)\
    selectivex_waitany(cnt, reqs, indx, stat)

#define MPI_Waitsome(incnt, reqs, outcnt, indices, stats)\
    selectivex_waitsome(incnt,reqs,outcnt,indices,stats)

#define MPI_Waitall(cnt, reqs, stats)\
    selectivex_waitall(cnt,reqs,stats)

#define MPI_Test(req, flag, st)\
    selectivex_test(req,flag,st)

#define MPI_Testany(cnt, reqs, indx, flag, st)\
    selectivex_testany(cnt,reqs,indx,flag,st)

#define MPI_Testsome(incnt, reqs, outcnt, indices, stats)\
    selectivex_testsome(incnt,reqs,outcnt,indices,stats)

#define MPI_Testall(cnt, reqs, flag, stats)\
    selectivex_testall(cnt,reqs,flag,stats)

#endif /*__SELECTIVEX__MPI_H_*/
