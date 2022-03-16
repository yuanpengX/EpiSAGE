# basic parameters
t=$1
prefix=$2
#datadir="/data2/xiongyuanpeng/gene_expression/data/kidney/"
mkdir "${datadir}/processed"
outfile="${datadir}/processed/splited_histone_modification/"
mkdir ${outfile}
mkdir "${datadir}/processed/feature/"
mkdir "${datadir}/processed/feature/histone/"
file="${datadir}/processed/${t}.bed"

# convert bigWig2bed
#source activate hicpro
echo "bigWigTowig";
bigWigToWig  ${datadir}/${prefix}*.bigWig ${datadir}/processed/${t}.wig ;
echo "wig2bed";
wig2bed < ${datadir}/processed/${t}.wig > ${datadir}/processed/${t}.bed --zero-indexed ;
echo "convert histone data ${t} ${prefix} done" ;