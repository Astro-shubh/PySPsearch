##########    Header information    #################
def dm_to_delay(dm, fmin, fmax):
    ## Returns delay in seconds corresponding to input dm and frequencies
    return (1/241.0)*dm*(1/(fmin*fmin) - 1.0/(fmax*fmax))

class headinfo:
    ## Reads the input filebank file header
	def __init__(self, filename):
		fil1 = sigpyproc.readers.FilReader(filename)
		self.nsamp = fil1.header.nsamples
		self.tsamp = fil1.header.tsamp        ## seconds
		self.fmax = fil1.header.fmax/1000.0    ## GHz
		self.fmin = fil1.header.fmin/1000.0    ## GHz
		self.nchan = fil1.header.nchans
		self.basename = fil1.header.basename

def generate_dmplan(header, dmlo, dmhi):
    ## Generates DM plan to be used in pyfdmt
	dm_plan = []
    down_plan = []
	dm_const = 1/241.0
	nu1 = header.fmin
	nu2 = header.fmax
	del_t = header.tsamp
	chan_width = (nu2 - nu1)/header.nchan
	min_delay = 2*dm_const*lodm*chan_width/(nu1)**3.0
	print(f" Minimum delay {min_delay} for dm {lodm}")
	down_factor = np.ceil(min_delay/del_t)
	del_t = del_t*down_factor
	max_delay = 2*dm_const*hidm*chan_width/(nu1)**3.0
	if(max_delay <= del_t):
		dm_plan.append([lodm, hidm])
        down_plan.append(down_factor)
		return dm_plan
	else:
		dm1 = lodm
		while(del_t < max_delay):
			dm2 = (del_t)/(2*dm_const*chan_width/(nu1)**3.0)
			if(dm2 >= hidm):
				dm_plan.append([dm1, hidm])
                down_plan.append(down_factor)
				break;
			down_factor = down_factor*2
			del_t = del_t*2
			dm_plan.append([dm1, dm2])
			dm1 = dm2
		dm_plan.append([dm1, hidm])
        down_plan.append(down_factor)
		return dm_plan, down_plan


def get_filters(max_width, tsamp):
    max_filt_size = max_width/tsamp
    num_filters = int(np.log2(max_filt_size))+1
    for i in range(fun_filters+1):
        filters.append(2**i)
    return filters

########## lists to store the detections   ############

def search_fil(filename, fmin, fmax, dmlo, dmhi, max_width):
    cand_Time = []
    cand_DM = []
    cand_Width = []
    cand_SNR = []
###  Header class to store header of filterbank  ########
    header = headinfo(filename)
    tsamp1 = header.tsamp
    nsamp = header.nsamp
    nchan = header.nchan
    fmin = header.fmin
    fmax = header.fmax

#########   DM plan without dm step, the DM step is set by pyfdmt algortihm based on the frequency range and time resolution   ################

    dm_plan, down_plan = generate_dmplan(header, lodm, hidm)
    print(dm_plan, down_plan)

#########    Filter block reading     ##################

    overlap = dm_to_delay(hidm, fmin, fmax) ## (1/241.0)*hidm.0*(1/(fmin*fmin) - 1.0/(fmax*fmax))  ## seconds

    print(overlap)

    overlap_samps = int(overlap/tsamp1)

    buffer_size = 131072                    ### Fixed to Cheetah buffer size

    iterations = int(np.floor(float(nsamp - overlap_samps)/float(buffer_size - overlap_samps))+1)

    print("number of interations : "+str(iterations))
    print(f"Low and high frequency {fmin} {fmax}")
    start = 0
    fil = sigpyproc.readers.FilReader(filename)
#####################    Going through block iterations    ########################
    for itr in range(iterations):
        if( (nsamp - start) < buffer_size):
            buffer_size = nsamp - start

        new_buffer = fil.read_block(start, buffer_size)
        start_time = start*tsamp1


    #################   going through the DM plan     ###########################
        for plan_number in range(len(dm_plan)):
            print("working on DM range: "+str(dm_plan[plan_number])+'\n')
            down_factor = down_plan[plan_number]
            using_buffer = new_buffer.downsample(1, down_factor)
            filters = get_filters(max_width, using_buffer.header.tsamp)

        ###########    Getting the block parameters    ##############
            tsamp = using_buffer.header.tsamp
            dm_time = transform(using_buffer.data, using_buffer.header.fmax, using_buffer.header.fmin, using_buffer.header.tsamp, dm_plan[plan_number][0], dm_plan[plan_number][1])
            DT_data = dm_time.data
            dm_list = dm_time.dms
            for dm_num in range(len(dm_list)):
                Tseries = DT_data[dm_num]
                rmed_width = 2.0*max_width/using_buffer.header.tsamp
                rmed_width = 2*int(rmed_width/2.0)+1
                T, D, W, S = modules.detection.detection(Tseries, filters, rmed_width, threshold, using_buffer.header.tsamp, dm_list[dm_num])
                for i in range(len(T)):
                    cand_Time.append(T[i]+start_time)
                    cand_DM.append(D[i])
                    cand_Width.append(W[i])
                    cand_SNR.append(S[i])
    #######  Setting the next start of the block_read  ##################
        start = start+buffer_size - overlap_samps
    return cand_Time, cand_DM, cand_Width, cand_SNR
