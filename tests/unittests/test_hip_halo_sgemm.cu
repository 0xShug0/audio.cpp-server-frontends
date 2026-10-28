#include "sgemm.cuh"
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#ifdef F16RAW
#include "convert.cuh"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <tuple>
#include <vector>

#define CHECK(x) do { const auto status=(x); if (status != 0) { std::fprintf(stderr, "API failure %s:%d: %s status=%d\n", __FILE__, __LINE__, #x, int(status)); std::exit(1); } } while (0)

static void run(int m, int n, int k, int pad, int tile_index = -1, bool plans = false, bool blas = false, int lowp = 0) {
    const bool vector_lds=tile_index == 4;
    const bool library_candidate=plans || blas || lowp != 0;
    const bool three_modes=vector_lds || library_candidate;
    const int candidate_mode=three_modes ? 2 : 1;
    const int lda=k+pad, ldb=(pad == 0 ? k : k+pad+1), ldc=m+pad;
    constexpr size_t guard = 64;  // 256-byte alignment for both implementations.
    constexpr float sentinel = -1234567.0f;
    const size_t a_elements=size_t(lda)*m, b_elements=size_t(ldb)*n, c_elements=size_t(ldc)*n;
    std::vector<float> a(a_elements+2*guard,sentinel), b(b_elements+2*guard,sentinel);
    std::vector<float> out(c_elements+2*guard,sentinel), ref(out.size(),sentinel);
    for (size_t i=0;i<a_elements;++i) a[guard+i]=float(int((i*17+13)%251)-125)/127.13f;
    for (size_t i=0;i<b_elements;++i) b[guard+i]=float(int((i*31+7)%257)-128)/131.73f;
    float *da_base,*db_base,*dc_base;
    CHECK(hipMalloc(&da_base,a.size()*4)); CHECK(hipMalloc(&db_base,b.size()*4)); CHECK(hipMalloc(&dc_base,out.size()*4));
    float *da=da_base+guard,*db=db_base+guard,*dc=dc_base+guard;
    CHECK(hipMemcpy(da_base,a.data(),a.size()*4,hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db_base,b.data(),b.size()*4,hipMemcpyHostToDevice));
    hipblasLtHandle_t handle; hipblasLtMatmulDesc_t desc;
    hipblasLtMatrixLayout_t la,lb,lc; hipblasLtMatmulPreference_t pref;
    CHECK(hipblasLtCreate(&handle)); CHECK(hipblasLtMatmulDescCreate(&desc,HIPBLAS_COMPUTE_32F,HIP_R_32F));
    hipblasOperation_t ta=HIPBLAS_OP_T;
    CHECK(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
    CHECK(hipblasLtMatrixLayoutCreate(&la,HIP_R_32F,k,m,lda));
    CHECK(hipblasLtMatrixLayoutCreate(&lb,HIP_R_32F,k,n,ldb));
    CHECK(hipblasLtMatrixLayoutCreate(&lc,HIP_R_32F,m,n,ldc));
    CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    size_t worksize=32*1024*1024;void *work;
    CHECK(hipMalloc(&work,worksize));
    CHECK(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&worksize,sizeof(worksize)));
    hipblasLtMatmulHeuristicResult_t algo;int count=0;
    CHECK(hipblasLtMatmulAlgoGetHeuristic(handle,desc,la,lb,lc,lc,pref,1,&algo,&count));
    if (!count) { std::fprintf(stderr,"SGEMM_PLAN_ERROR,algo_index=-1,status=1,stage=first1_count,count=0\n"); std::exit(1); }
    hipblasLtMatmulHeuristicResult_t alternatives[8]; int plan_count=1, plan_index=0;
    hipblasHandle_t blas_handle=nullptr;
    if (blas) {
        CHECK(hipblasCreate(&blas_handle));
        CHECK(hipblasSetPointerMode(blas_handle,HIPBLAS_POINTER_MODE_HOST));
        CHECK(hipblasSetStream(blas_handle,nullptr));
    }
    float alpha=1,beta=0;
#ifdef F16RAW
    constexpr size_t low_guard=128;
    const size_t low_a_elements=size_t(k)*m, low_b_elements=size_t(k)*n;
    uint16_t *low_a_base=nullptr,*low_b_base=nullptr;
    hipblasLtMatrixLayout_t low_la=nullptr,low_lb=nullptr;
    hipblasLtMatmulHeuristicResult_t low_algo;
    to_fp16_nc_cuda_t to_f16=nullptr;
    to_bf16_nc_cuda_t to_bf16=nullptr;
    size_t scratch_bytes=0;
    if (lowp) {
        if (lowp == 1) to_f16=ggml_get_to_fp16_nc_cuda(GGML_TYPE_F32);
        else to_bf16=ggml_get_to_bf16_nc_cuda(GGML_TYPE_F32);
        if (!(lowp == 1 ? bool(to_f16) : bool(to_bf16))) {
            std::fprintf(stderr,"SGEMM_LOWP_ERROR,dtype=%s,stage=missing_converter\n",lowp == 1 ? "f16" : "bf16"); std::exit(1);
        }
        scratch_bytes=(low_a_elements+low_b_elements+4*low_guard)*sizeof(uint16_t);
        CHECK(hipMalloc(&low_a_base,(low_a_elements+2*low_guard)*sizeof(uint16_t)));
        CHECK(hipMalloc(&low_b_base,(low_b_elements+2*low_guard)*sizeof(uint16_t)));
        CHECK(hipMemset(low_a_base,0xa5,(low_a_elements+2*low_guard)*sizeof(uint16_t)));
        CHECK(hipMemset(low_b_base,0xa5,(low_b_elements+2*low_guard)*sizeof(uint16_t)));
        const hipDataType input_type=lowp == 1 ? HIP_R_16F : HIP_R_16BF;
        CHECK(hipblasLtMatrixLayoutCreate(&low_la,input_type,k,m,k));
        CHECK(hipblasLtMatrixLayoutCreate(&low_lb,input_type,k,n,k));
        int low_count=0;
        CHECK(hipblasLtMatmulAlgoGetHeuristic(handle,desc,low_la,low_lb,lc,lc,pref,1,&low_algo,&low_count));
        if (!low_count || low_algo.state != HIPBLAS_STATUS_SUCCESS || low_algo.workspaceSize > worksize) {
            std::fprintf(stderr,"SGEMM_LOWP_ERROR,dtype=%s,stage=heuristic,count=%d,status=%d\n",lowp == 1 ? "f16" : "bf16",low_count,low_count ? int(low_algo.state) : -1); std::exit(1);
        }
    }
#endif
    auto launch=[&](int mode) {
#ifdef F16RAW
        // Mode 2 always converts both inputs; modes 3/4 diagnose conversion/GEMM separately.
        if (lowp && (mode == 2 || mode == 3)) {
            if (lowp == 1) {
                to_f16(da,reinterpret_cast<half *>(low_a_base+low_guard),k,m,1,1,lda,0,0,nullptr);
                to_f16(db,reinterpret_cast<half *>(low_b_base+low_guard),k,n,1,1,ldb,0,0,nullptr);
            } else {
                to_bf16(da,reinterpret_cast<nv_bfloat16 *>(low_a_base+low_guard),k,m,1,1,lda,0,0,nullptr);
                to_bf16(db,reinterpret_cast<nv_bfloat16 *>(low_b_base+low_guard),k,n,1,1,ldb,0,0,nullptr);
            }
            if (mode == 3) return;
        }
        if (lowp && (mode == 2 || mode == 4)) {
            CHECK(hipblasLtMatmul(handle,desc,&alpha,low_a_base+low_guard,low_la,low_b_base+low_guard,low_lb,&beta,dc,lc,dc,lc,&low_algo.algo,work,worksize,nullptr));
            return;
        }
#endif
        if (blas && mode == 2) {
            const auto status=hipblasSgemm(blas_handle,HIPBLAS_OP_T,HIPBLAS_OP_N,m,n,k,&alpha,da,lda,db,ldb,&beta,dc,ldc);
            if (status != HIPBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr,"SGEMM_BLAS_ERROR,status=%d,stage=hipblasSgemm\n",int(status)); std::exit(1);
            }
        } else if (plans && mode == 2) {
            const auto status=hipblasLtMatmul(handle,desc,&alpha,da,la,db,lb,&beta,dc,lc,dc,lc,&alternatives[plan_index].algo,work,worksize,nullptr);
            if (status != HIPBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr,"SGEMM_PLAN_ERROR,algo_index=%d,status=%d,stage=matmul\n",plan_index,int(status)); std::exit(1);
            }
        } else if (mode != 0) {
            if (tile_index < 0 || (vector_lds && mode == 1)) ggml_hip_halo_sgemm(da,db,dc,m,n,k,lda,ldb,ldc,0,0,0,1,nullptr);
            else ggml_hip_halo_sgemm(da,db,dc,m,n,k,lda,ldb,ldc,0,0,0,1,nullptr,tile_index);
        }
        else CHECK(hipblasLtMatmul(handle,desc,&alpha,da,la,db,lb,&beta,dc,lc,dc,lc,&algo.algo,work,worksize,nullptr));
    };
    auto check_output=[&](const std::vector<float> & values, const char * label) {
        for (size_t i=0;i<guard;++i) {
            if (values[i]!=sentinel || values[guard+c_elements+i]!=sentinel) {
                std::fprintf(stderr,"%s output boundary guard failed: tile=%d offset=%zu\n",label,tile_index,i);
                std::exit(4);
            }
        }
        for (int j=0;j<n;++j) for (int i=m;i<ldc;++i) {
            if (values[guard+size_t(j)*ldc+i]!=sentinel) {
                std::fprintf(stderr,"%s output row padding failed: tile=%d row=%d col=%d\n",label,tile_index,i,j);
                std::exit(4);
            }
        }
    };
    auto check_input_guards=[&](const float * base, size_t elements, const char * label) {
        float edges[2*guard];
        CHECK(hipMemcpy(edges,base,guard*4,hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(edges+guard,base+guard+elements,guard*4,hipMemcpyDeviceToHost));
        for (size_t i=0;i<2*guard;++i) {
            if (edges[i]!=sentinel) {
                std::fprintf(stderr,"%s input boundary guard failed: tile=%d offset=%zu\n",label,tile_index,i);
                std::exit(4);
            }
        }
    };
    // Reset all C storage before each correctness launch, so a preceding
    // implementation cannot supply valid outputs or hide padding writes.
    CHECK(hipMemcpy(dc_base,ref.data(),ref.size()*4,hipMemcpyHostToDevice));
    launch(0);CHECK(hipGetLastError());
    CHECK(hipMemcpy(ref.data(),dc_base,ref.size()*4,hipMemcpyDeviceToHost));
    check_output(ref,"hipBLASLt");
    std::vector<float> reference64;
    if (vector_lds) {
        reference64.resize(out.size());
        CHECK(hipMemcpy(dc_base,out.data(),out.size()*4,hipMemcpyHostToDevice));
        launch(1);CHECK(hipGetLastError());
        CHECK(hipMemcpy(reference64.data(),dc_base,reference64.size()*4,hipMemcpyDeviceToHost));
        check_output(reference64,"baseline64");
    }
    CHECK(hipMemcpy(dc_base,out.data(),out.size()*4,hipMemcpyHostToDevice));
    launch(library_candidate ? 1 : candidate_mode);CHECK(hipGetLastError());
    CHECK(hipMemcpy(out.data(),dc_base,out.size()*4,hipMemcpyDeviceToHost));
    check_output(out,"halo");
    check_input_guards(da_base,a_elements,"A");
    check_input_guards(db_base,b_elements,"B");
    if (vector_lds) {
        for (int j=0;j<n;++j) {
            const size_t offset=guard+size_t(j)*ldc;
            if (std::memcmp(out.data()+offset,reference64.data()+offset,size_t(m)*sizeof(float))!=0) {
                std::fprintf(stderr,"vector LDS differs bitwise from baseline64: col=%d\n",j);
                std::exit(5);
            }
        }
    }
    // Independent double accumulation at distributed output positions, plus
    // exhaustive comparison with the existing FP32 library implementation.
    for (int j=0;j<n;++j) for(int i=0;i<m;++i) {
        float value=out[guard+size_t(j)*ldc+i], expected=ref[guard+size_t(j)*ldc+i];
        if (!std::isfinite(value) || !std::isfinite(expected) || std::abs(value-expected)>2e-4f+2e-5f*std::abs(expected)) std::exit(2);
        if ((size_t(j)*m+i)%std::max<size_t>(1,size_t(m)*n/257)==0) {
            double exact=0;for(int q=0;q<k;++q) exact+=double(a[guard+size_t(i)*lda+q])*b[guard+size_t(j)*ldb+q];
            if (std::abs(value-exact)>2e-4+2e-5*std::abs(exact) ||
                std::abs(expected-exact)>2e-4+2e-5*std::abs(exact)) std::exit(3);
        }
    }
    if (library_candidate) reference64=out;
    if (plans) {
        const auto status=hipblasLtMatmulAlgoGetHeuristic(handle,desc,la,lb,lc,lc,pref,8,alternatives,&plan_count);
        if (status != HIPBLAS_STATUS_SUCCESS) { std::fprintf(stderr,"SGEMM_PLAN_ERROR,algo_index=-1,status=%d,stage=plans_query\n",int(status)); std::exit(1); }
        std::printf("SGEMM_PLANS,m=%d,n=%d,k=%d,pad=%d,count=%d,workspace_cap_bytes=%zu,compute=32F,storage=R32F\n",m,n,k,pad,plan_count,worksize);
        if (!plan_count) { std::fprintf(stderr,"SGEMM_PLAN_ERROR,algo_index=-1,status=1,stage=plans_count,count=0\n"); std::exit(1); }
    }
    for (plan_index=0;plan_index<plan_count;++plan_index) {
        double nmse=0,max_abs=0,cosine=0;
#ifdef F16RAW
        double double_max_abs=0,quantized_double_max_abs=0;
#endif
        const char * result_tag=lowp ? "SGEMM_LOWP" : blas ? "SGEMM_BLAS" : "SGEMM_PLAN";
        if (plans) {
            std::printf("SGEMM_PLAN_BEGIN,algo_index=%d,status=%d\n",plan_index,int(alternatives[plan_index].state));
            if (alternatives[plan_index].state != HIPBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr,"SGEMM_PLAN_ERROR,algo_index=%d,status=%d,stage=heuristic_state\n",plan_index,int(alternatives[plan_index].state)); std::exit(1);
            }
            if (alternatives[plan_index].workspaceSize > worksize) {
                std::fprintf(stderr,"SGEMM_PLAN_ERROR,algo_index=%d,status=1,stage=workspace_cap\n",plan_index); std::exit(1);
            }
        }
        if (library_candidate) {
            std::fill(out.begin(),out.end(),sentinel);
            CHECK(hipMemcpy(dc_base,out.data(),out.size()*4,hipMemcpyHostToDevice));
            launch(2);
            const auto status=hipMemcpy(out.data(),dc_base,out.size()*4,hipMemcpyDeviceToHost);
            if (status != hipSuccess) { std::fprintf(stderr,"%s_ERROR,algo_index=%d,status=%d,stage=correctness_sync\n",result_tag,plan_index,int(status)); std::exit(1); }
            char label[64]; std::snprintf(label,sizeof(label),"%s algo_index=%d status=4",result_tag,plan_index);
            check_output(out,label); check_input_guards(da_base,a_elements,label); check_input_guards(db_base,b_elements,label);
#ifdef F16RAW
            std::vector<uint16_t> low_a_host,low_b_host;
            if (lowp) {
                low_a_host.resize(low_a_elements+2*low_guard); low_b_host.resize(low_b_elements+2*low_guard);
                CHECK(hipMemcpy(low_a_host.data(),low_a_base,low_a_host.size()*sizeof(uint16_t),hipMemcpyDeviceToHost));
                CHECK(hipMemcpy(low_b_host.data(),low_b_base,low_b_host.size()*sizeof(uint16_t),hipMemcpyDeviceToHost));
                for (int operand=0;operand<2;++operand) {
                    const auto & packed=operand ? low_b_host : low_a_host;
                    const auto & source=operand ? b : a;
                    const int rows=operand ? n : m, stride=operand ? ldb : lda;
                    for (size_t i=0;i<low_guard;++i) {
                        if (packed[i]!=0xa5a5 || packed[packed.size()-low_guard+i]!=0xa5a5) {
                            std::fprintf(stderr,"SGEMM_LOWP_ERROR,stage=conversion_guard,operand=%d\n",operand); std::exit(4);
                        }
                    }
                    for (int row=0;row<rows;++row) for (int q=0;q<k;++q) {
                        const uint16_t bits=packed[low_guard+size_t(row)*k+q];
                        const uint16_t exponent=lowp == 1 ? 0x7c00 : 0x7f80;
                        if ((bits & exponent)==exponent) {
                            std::fprintf(stderr,"SGEMM_LOWP_ERROR,stage=conversion_nonfinite,dtype=%s,operand=%d,row=%d,k=%d\n",lowp == 1 ? "f16" : "bf16",operand,row,q); std::exit(3);
                        }
                        const float value=source[guard+size_t(row)*stride+q];
                        const uint16_t expected=lowp == 1 ? ggml_fp32_to_fp16(value) : ggml_fp32_to_bf16(value).bits;
                        if (!std::isfinite(value) || bits!=expected) {
                            std::fprintf(stderr,"SGEMM_LOWP_ERROR,stage=conversion_mismatch,operand=%d,row=%d,k=%d\n",operand,row,q); std::exit(3);
                        }
                    }
                }
            }
#endif
            double error2=0,reference2=0,output2=0,dot=0; bool valid=true;
            for (int j=0;j<n;++j) for (int i=0;i<m;++i) {
                const size_t offset=guard+size_t(j)*ldc+i;
                const double value=out[offset], baseline=reference64[offset], delta=value-baseline;
                valid &= std::isfinite(value) && (lowp || std::abs(value-ref[offset])<=2e-4+2e-5*std::abs(double(ref[offset])));
                error2+=delta*delta; reference2+=baseline*baseline; output2+=value*value; dot+=value*baseline;
                max_abs=std::max(max_abs,std::abs(delta));
                if ((size_t(j)*m+i)%std::max<size_t>(1,size_t(m)*n/257)==0) {
                    double exact=0; for (int q=0;q<k;++q) exact+=double(a[guard+size_t(i)*lda+q])*b[guard+size_t(j)*ldb+q];
#ifdef F16RAW
                    double_max_abs=std::max(double_max_abs,std::abs(value-exact));
                    if (lowp) {
                        double quantized=0;
                        for (int q=0;q<k;++q) {
                            const uint16_t aa=low_a_host[low_guard+size_t(i)*k+q], bb=low_b_host[low_guard+size_t(j)*k+q];
                            const double av=lowp == 1 ? ggml_fp16_to_fp32(aa) : ggml_bf16_to_fp32(ggml_bf16_t{aa});
                            const double bv=lowp == 1 ? ggml_fp16_to_fp32(bb) : ggml_bf16_to_fp32(ggml_bf16_t{bb});
                            quantized+=av*bv;
                        }
                        quantized_double_max_abs=std::max(quantized_double_max_abs,std::abs(value-quantized));
                        valid &= std::abs(value-quantized)<=2e-4+2e-5*std::abs(quantized);
                    } else
#endif
                    valid &= std::abs(value-exact)<=2e-4+2e-5*std::abs(exact);
                }
            }
            nmse=error2/reference2; cosine=dot/std::sqrt(reference2*output2);
            valid &= std::isfinite(nmse) && std::isfinite(cosine);
            if (lowp) valid &= nmse <= (lowp == 1 ? 1e-6 : 1e-4);
            if (!valid) {
                std::fprintf(stderr,"%s_ERROR,algo_index=%d,status=2,stage=numerics,nmse=%.12g,max_abs=%.12g,cosine=%.12g\n",result_tag,plan_index,nmse,max_abs,cosine); std::exit(2);
            }
        }
        if (tile_index >= 0) {
            const int tile_m=(tile_index & 2) ? 128 : 64, tile_n=(tile_index & 1) ? 128 : 64;
            const int pitch_a=vector_lds ? 68 : tile_m+1, pitch_b=vector_lds ? 68 : tile_n+1;
            const size_t shared_bytes=32*size_t(pitch_a+pitch_b)*sizeof(float);
            std::printf("SGEMM_TILE,index=%d,m=%d,n=%d,threads=256,static_shared_bytes=%zu\n",
                        tile_index,tile_m,tile_n,shared_bytes);
        }
        hipEvent_t start,end;CHECK(hipEventCreate(&start));CHECK(hipEventCreate(&end));
        std::vector<float> times[3];
        // Cover all six permutations for each three-mode comparison:
        // every mode occupies every position twice, with balanced pairwise order.
        constexpr int orders[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
        const int calls=library_candidate ? 4 : 8;
        for(int round=0;round<(three_modes ? 6 : 7);++round) for(int order=0;order<(three_modes ? 3 : 2);++order) {
            int mode=three_modes ? orders[round][order] : (round+order)%2;
            for(int i=0;i<3;++i)launch(mode);
            CHECK(hipEventRecord(start));for(int i=0;i<calls;++i)launch(mode);
            CHECK(hipEventRecord(end));CHECK(hipEventSynchronize(end));
            float ms;CHECK(hipEventElapsedTime(&ms,start,end));times[mode].push_back(ms/calls);
        }
        float medians[3] = {};
        for (int mode=0;mode<=candidate_mode;++mode) {
            auto & t=times[mode];
            std::sort(t.begin(),t.end());
            const size_t mid=t.size()/2;
            medians[mode]=t.size()%2 ? t[mid] : (t[mid-1]+t[mid])*0.5f;
        }
#ifdef F16RAW
        float conversion_ms=0,gemm_ms=0;
        if (lowp) {
            std::vector<float> stage_times[2];
            for (int round=0;round<6;++round) for (int order=0;order<2;++order) {
                const int stage=(round+order)%2, mode=stage == 0 ? 3 : 4;
                for (int i=0;i<3;++i) launch(mode);
                CHECK(hipEventRecord(start)); for (int i=0;i<calls;++i) launch(mode);
                CHECK(hipEventRecord(end)); CHECK(hipEventSynchronize(end));
                float ms; CHECK(hipEventElapsedTime(&ms,start,end)); stage_times[stage].push_back(ms/calls);
            }
            for (auto & t:stage_times) std::sort(t.begin(),t.end());
            conversion_ms=(stage_times[0][2]+stage_times[0][3])*0.5f;
            gemm_ms=(stage_times[1][2]+stage_times[1][3])*0.5f;
            for (int operand=0;operand<2;++operand) {
                const auto * base=operand ? low_b_base : low_a_base;
                const size_t elements=operand ? low_b_elements : low_a_elements;
                uint16_t edges[2*low_guard];
                CHECK(hipMemcpy(edges,base,low_guard*sizeof(uint16_t),hipMemcpyDeviceToHost));
                CHECK(hipMemcpy(edges+low_guard,base+low_guard+elements,low_guard*sizeof(uint16_t),hipMemcpyDeviceToHost));
                for (size_t i=0;i<2*low_guard;++i) if (edges[i]!=0xa5a5) {
                    std::fprintf(stderr,"SGEMM_LOWP_ERROR,stage=timed_conversion_guard,operand=%d\n",operand); std::exit(4);
                }
            }
        }
#endif
        CHECK(hipGetLastError());
        CHECK(hipMemcpy(out.data(),dc_base,out.size()*4,hipMemcpyDeviceToHost));
        check_output(out,"timed output");
        check_input_guards(da_base,a_elements,"A");
        check_input_guards(db_base,b_elements,"B");
        const float library_ms=medians[0], candidate_ms=medians[candidate_mode];
#ifdef F16RAW
        if (lowp) {
            const size_t payload_bytes=(low_a_elements+low_b_elements)*sizeof(uint16_t);
            const size_t explicit_device_bytes=(a.size()+b.size()+out.size())*sizeof(float)+worksize+scratch_bytes;
            std::printf("SGEMM_LOWP,dtype=%s,m=%d,n=%d,k=%d,pad=%d,first1_ms=%.6f,reference64_ms=%.6f,candidate_ms=%.6f,conversion_ms=%.6f,gemm_ms=%.6f,speedup_vs_first1=%.4f,speedup_vs_reference64=%.4f,scratch_payload_bytes=%zu,scratch_bytes=%zu,explicit_device_bytes=%zu,nmse_limit=%.12g,nmse=%.12g,max_abs=%.12g,cosine=%.12g,double_max_abs=%.12g,quantized_double_max_abs=%.12g,conversions_per_call=2\n",
                        lowp == 1 ? "f16" : "bf16",m,n,k,pad,library_ms,medians[1],candidate_ms,conversion_ms,gemm_ms,library_ms/candidate_ms,medians[1]/candidate_ms,
                        payload_bytes,scratch_bytes,explicit_device_bytes,lowp == 1 ? 1e-6 : 1e-4,nmse,max_abs,cosine,double_max_abs,quantized_double_max_abs);
        } else
#endif
        if (blas) {
            std::printf("SGEMM_BLAS,m=%d,n=%d,k=%d,pad=%d,first1_ms=%.6f,reference64_ms=%.6f,blas_ms=%.6f,speedup_vs_first1=%.4f,speedup_vs_reference64=%.4f,nmse=%.12g,max_abs=%.12g,cosine=%.12g\n",
                        m,n,k,pad,library_ms,medians[1],candidate_ms,library_ms/candidate_ms,medians[1]/candidate_ms,nmse,max_abs,cosine);
        } else if (plans) {
            std::printf("SGEMM_PLAN,algo_index=%d,workspace_bytes=%zu,first1_ms=%.6f,reference64_ms=%.6f,candidate_ms=%.6f,speedup_vs_first1=%.4f,speedup_vs_reference64=%.4f,nmse=%.12g,max_abs=%.12g,cosine=%.12g\n",
                        plan_index,alternatives[plan_index].workspaceSize,library_ms,medians[1],candidate_ms,library_ms/candidate_ms,medians[1]/candidate_ms,nmse,max_abs,cosine);
        } else {
        std::printf("SGEMM,%d,%d,%d,%d,%.6f,%.6f,%.4f\n",m,n,k,pad,library_ms,candidate_ms,library_ms/candidate_ms);
        if (vector_lds) {
            const float reference64_ms=medians[1];
            std::printf("SGEMM_LDSVEC,m=%d,n=%d,k=%d,pad=%d,reference64_ms=%.6f,candidate_ms=%.6f,speedup_vs_reference64=%.4f,bitwise=pass\n",
                        m,n,k,pad,reference64_ms,candidate_ms,reference64_ms/candidate_ms);
        }
        }
        CHECK(hipEventDestroy(start));CHECK(hipEventDestroy(end));
    }
    CHECK(hipblasLtMatmulPreferenceDestroy(pref));CHECK(hipblasLtMatrixLayoutDestroy(la));
    CHECK(hipblasLtMatrixLayoutDestroy(lb));CHECK(hipblasLtMatrixLayoutDestroy(lc));
    CHECK(hipblasLtMatmulDescDestroy(desc));CHECK(hipblasLtDestroy(handle));
#ifdef F16RAW
    if (lowp) {
        CHECK(hipblasLtMatrixLayoutDestroy(low_la)); CHECK(hipblasLtMatrixLayoutDestroy(low_lb));
        CHECK(hipFree(low_a_base)); CHECK(hipFree(low_b_base));
    }
#endif
    if (blas) CHECK(hipblasDestroy(blas_handle));
    CHECK(hipFree(work));CHECK(hipFree(da_base));CHECK(hipFree(db_base));CHECK(hipFree(dc_base));
}

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc != 1 && argc != 2 && argc != 5 && argc != 6) {
        std::fprintf(stderr,"usage: %s [shape_log | m n k pad [tile_index:0..4|plans|blas|f16|bf16]] (f16/bf16 require F16RAW)\n",argv[0]);
        return 1;
    }
    int devices=0;
    if (hipGetDeviceCount(&devices)!=hipSuccess || devices==0)return 77;
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props,0));
    if (std::string(props.gcnArchName).find("gfx1151")!=0)return 77;
    if (argc == 5 || argc == 6) {
        const int m = std::stoi(argv[1]), n = std::stoi(argv[2]);
        const int k = std::stoi(argv[3]), pad = std::stoi(argv[4]);
        size_t parsed=0;
        const bool plans=argc == 6 && std::string(argv[5]) == "plans";
        const bool blas=argc == 6 && std::string(argv[5]) == "blas";
        const int lowp=argc == 6 ? (std::string(argv[5]) == "f16" ? 1 : std::string(argv[5]) == "bf16" ? 2 : 0) : 0;
#ifndef F16RAW
        if (lowp) { std::fprintf(stderr,"f16/bf16 require a -DF16RAW build\n"); return 1; }
#endif
        const int tile_index=argc == 6 && !plans && !blas && !lowp ? std::stoi(argv[5],&parsed) : -1;
        if (m <= 0 || n <= 0 || k <= 0 || pad < 0) return 1;
        if (argc == 6 && !plans && !blas && !lowp && (tile_index < 0 || tile_index > 4 || parsed != std::string(argv[5]).size())) return 1;
        run(m, n, k, pad, tile_index, plans, blas, lowp);
        std::puts("PASS selected SGEMM shape");
        return 0;
    }
    if (argc == 2) {
        std::ifstream file(argv[1]);
        if (!file) return 1;
        std::string line;
        std::set<std::tuple<int,int,int>> shapes;
        while (std::getline(file,line)) {
            auto pos=line.find("[GEMM_SHAPE] ");
            if(pos==std::string::npos)continue;
            std::istringstream in(line.substr(pos+13));
            int m,n,k,ta,tb,tc,lda,ldb,ldc,batches;
            if (!(in>>m>>n>>k>>ta>>tb>>tc>>lda>>ldb>>ldc>>batches))return 1;
            if (ta==0 && tb==0 && tc==0 && batches==1)shapes.emplace(m,n,k);
        }
        if(shapes.empty())return 1;
        for(auto s:shapes)run(std::get<0>(s),std::get<1>(s),std::get<2>(s),0);
        std::puts("PASS production SGEMM shapes");
        return 0;
    }
    for(int pad:{0,3})for(auto s:std::vector<std::vector<int>>{
        {1,1,1},{31,33,65},{64,64,64},{67,131,97},
        {512,128,1536},{256,1024,3584},{128,8192,1792},
        {64,32768,896},{1024,128,1024},{128,550,3200}})run(s[0],s[1],s[2],pad);
    std::puts("PASS SGEMM: library parity and independent double reference");
}
