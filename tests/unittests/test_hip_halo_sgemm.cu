#include "sgemm.cuh"
#include <hipblaslt/hipblaslt.h>
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

#define CHECK(x) do { if ((x) != 0) { std::fprintf(stderr, "API failure %s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)

static void run(int m, int n, int k, int pad, int tile_index = -1) {
    const bool vector_lds=tile_index == 4;
    const int candidate_mode=vector_lds ? 2 : 1;
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
    if (!count) std::exit(1);
    float alpha=1,beta=0;
    auto launch=[&](int mode) {
        if (mode != 0) {
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
    launch(candidate_mode);CHECK(hipGetLastError());
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
    if (tile_index >= 0) {
        const int tile_m=(tile_index & 2) ? 128 : 64, tile_n=(tile_index & 1) ? 128 : 64;
        const int pitch_a=vector_lds ? 68 : tile_m+1, pitch_b=vector_lds ? 68 : tile_n+1;
        const size_t shared_bytes=32*size_t(pitch_a+pitch_b)*sizeof(float);
        std::printf("SGEMM_TILE,index=%d,m=%d,n=%d,threads=256,static_shared_bytes=%zu\n",
                    tile_index,tile_m,tile_n,shared_bytes);
    }
    hipEvent_t start,end;CHECK(hipEventCreate(&start));CHECK(hipEventCreate(&end));
    std::vector<float> times[3];
    // For index 4, cover all six permutations of library/baseline64/vector LDS:
    // every mode occupies every position twice, with balanced pairwise order.
    constexpr int orders[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    for(int round=0;round<(vector_lds ? 6 : 7);++round) for(int order=0;order<(vector_lds ? 3 : 2);++order) {
        int mode=vector_lds ? orders[round][order] : (round+order)%2;
        for(int i=0;i<3;++i)launch(mode);
        CHECK(hipEventRecord(start));for(int i=0;i<8;++i)launch(mode);
        CHECK(hipEventRecord(end));CHECK(hipEventSynchronize(end));
        float ms;CHECK(hipEventElapsedTime(&ms,start,end));times[mode].push_back(ms/8);
    }
    float medians[3] = {};
    for (int mode=0;mode<=candidate_mode;++mode) {
        auto & t=times[mode];
        std::sort(t.begin(),t.end());
        const size_t mid=t.size()/2;
        medians[mode]=t.size()%2 ? t[mid] : (t[mid-1]+t[mid])*0.5f;
    }
    CHECK(hipGetLastError());
    CHECK(hipMemcpy(out.data(),dc_base,out.size()*4,hipMemcpyDeviceToHost));
    check_output(out,"timed output");
    check_input_guards(da_base,a_elements,"A");
    check_input_guards(db_base,b_elements,"B");
    const float library_ms=medians[0], candidate_ms=medians[candidate_mode];
    std::printf("SGEMM,%d,%d,%d,%d,%.6f,%.6f,%.4f\n",m,n,k,pad,library_ms,candidate_ms,library_ms/candidate_ms);
    if (vector_lds) {
        const float reference64_ms=medians[1];
        std::printf("SGEMM_LDSVEC,m=%d,n=%d,k=%d,pad=%d,reference64_ms=%.6f,candidate_ms=%.6f,speedup_vs_reference64=%.4f,bitwise=pass\n",
                    m,n,k,pad,reference64_ms,candidate_ms,reference64_ms/candidate_ms);
    }
    CHECK(hipEventDestroy(start));CHECK(hipEventDestroy(end));
    CHECK(hipblasLtMatmulPreferenceDestroy(pref));CHECK(hipblasLtMatrixLayoutDestroy(la));
    CHECK(hipblasLtMatrixLayoutDestroy(lb));CHECK(hipblasLtMatrixLayoutDestroy(lc));
    CHECK(hipblasLtMatmulDescDestroy(desc));CHECK(hipblasLtDestroy(handle));
    CHECK(hipFree(work));CHECK(hipFree(da_base));CHECK(hipFree(db_base));CHECK(hipFree(dc_base));
}

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc != 1 && argc != 2 && argc != 5 && argc != 6) {
        std::fprintf(stderr,"usage: %s [shape_log | m n k pad [tile_index:0..4]]\n",argv[0]);
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
        const int tile_index=argc == 6 ? std::stoi(argv[5],&parsed) : -1;
        if (m <= 0 || n <= 0 || k <= 0 || pad < 0) return 1;
        if (argc == 6 && (tile_index < 0 || tile_index > 4 || parsed != std::string(argv[5]).size())) return 1;
        run(m, n, k, pad, tile_index);
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
