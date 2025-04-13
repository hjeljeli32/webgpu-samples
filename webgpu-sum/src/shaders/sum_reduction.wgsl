@group(0) @binding(0) var<storage, read> inputBuffer: array<u32>;
@group(0) @binding(1) var<storage, read_write> resultBuffer: array<u32>;

// Shared memory for partial sums within a workgroup
var<workgroup> partialSums: array<u32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) group_id: vec3<u32>) {

    let globalIndex = global_id.x;
    let localIndex = local_id.x;

    // Step 1: Load input data into shared memory
    partialSums[localIndex] = inputBuffer[globalIndex];

    workgroupBarrier(); // Synchronize threads in workgroup

    // Step 2: Perform parallel reduction within the workgroup
    var stride = 32u;
    while (stride > 0) {
        if (localIndex < stride) {
            partialSums[localIndex] += partialSums[localIndex + stride];
        }
        stride /= 2u;
        workgroupBarrier();
    }

    // Step 3: Thread 0 writes the result of this workgroup to resultBuffer
    if (localIndex == 0u) {
        resultBuffer[group_id.x] = partialSums[0];
    }
}