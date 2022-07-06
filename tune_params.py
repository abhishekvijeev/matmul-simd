import subprocess

OUTPUT_FILE_PATH = "/home/ubuntu/hw1-pa1-avijeev-pgangwar/op_file.log"
op_file = open(OUTPUT_FILE_PATH, "a")
op_file.truncate(0)

KB = 2**10
MB = 2**20
GB = 2**30

L1_BYTES = 32 * KB
L2_BYTES = 256 * KB
L3_BYTES = 30 * MB
DOUBLE_SIZE = 8
L1_DOUBLES = L1_BYTES / DOUBLE_SIZE
L2_DOUBLES = L2_BYTES / DOUBLE_SIZE
L3_DOUBLES = L3_BYTES / DOUBLE_SIZE

#MIN_MR_SIZE = 4
#MAX_MR_SIZE = 64
MIN_MR_SIZE = 1 #Using log2(size) 
MAX_MR_SIZE = 6 #Using log2(size)

#MIN_NR_SIZE = 4
#MAX_NR_SIZE = 64
MIN_NR_SIZE = 1 #Using log2(size) 
MAX_NR_SIZE = 7 #Using log2(size)


DEFAULT_FLAGS = ' -O3 -march=core-avx2 -DOPENBLAS_SINGLETHREAD '

MR_VALUES = [4, 8, 16, 32]
NR_VALUES = [12, 16, 24, 32, 36, 48, 60, 64]
KC_VALUES = [32, 64, 128, 256]
MC_VALUES = [(L3_DOUBLES / kc) for kc in KC_VALUES]
#MC_VALUES = [16384, 8192, 4096, 2048, 1024, 512]
NC_VALUES = [(L2_DOUBLES / kc) for kc in KC_VALUES]
#for MR in range(MIN_MR_SIZE, MAX_MR_SIZE, MIN_MR_SIZE):
for i in range(len(MR_VALUES)):
    MR = int(MR_VALUES[i])
    #for NR in range(MIN_NR_SIZE, MAX_NR_SIZE, MIN_NR_SIZE):
    for j in range(len(NR_VALUES)):
        NR = int(NR_VALUES[i])
        for k in range(len(KC_VALUES)):
            KC = int(KC_VALUES[i])
            MC = int(MC_VALUES[i])
            NC = int(NC_VALUES[i])
            params = ' -DDGEMM_MC={} -DDGEMM_NC={} -DDGEMM_KC={} -DDGEMM_MR={} -DDGEMM_NR={} '
            MY_OPT = DEFAULT_FLAGS + params.format(MC, NC, KC, MR, NR)
            subprocess.call(['make', 'MY_OPT=' + MY_OPT])
            process = subprocess.Popen(['./benchmark-blislab'], stdout=subprocess.PIPE)
            #execution_output = str(process.communicate()[0]).split('\\n')[-2].strip() + '\n'
            #to_write = 'MC={} NC={} KC={} MR={} NR={} '.format(MC, NC, KC, MR, NR) + execution_output
            to_write = 'MC={} NC={} KC={} MR={} NR={}\n\n'.format(MC, NC, KC, MR, NR)
            op_file.write(to_write)
            execution_output_lines = str(process.communicate()[0]).strip().split('\\n')
            for line in execution_output_lines:
                if ("'" in line):
                    continue
                line = line.replace('\\t', ' ')
                op_file.write(line.strip() + '\n')
            #to_write = 'MC={} NC={} KC={} MR={} NR={} '.format(MC, NC, KC, MR, NR) + execution_output
            op_file.write("\n\n")
            op_file.flush()
op_file.close()
