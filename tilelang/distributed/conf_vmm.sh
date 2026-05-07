MAJOR=$(grep nvidia-caps-imex-channels /proc/devices | awk '{print $1}')

# Use mknod to open fabric channel 0
IMAX_CHANNEL_PATH=/dev/nvidia-caps-imex-channels/
IMAX_CHANNEL_NAME=channel0
sudo mkdir -p $IMAX_CHANNEL_PATH
sudo mknod $IMAX_CHANNEL_PATH/$IMAX_CHANNEL_NAME c $MAJOR 0
sudo chmod 666 $IMAX_CHANNEL_PATH/$IMAX_CHANNEL_NAME

# Validate the channel is created
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
if [ ! -c $IMAX_CHANNEL_PATH/$IMAX_CHANNEL_NAME ]; then
    echo -e "${RED}Failed to create channel $IMAX_CHANNEL_NAME${NC}"
    exit 1
else
    echo -e "${GREEN}Channel $IMAX_CHANNEL_NAME created successfully${NC}"
fi

# Optional: set environment variables to use VMM and distributed
export TILESCALE_USE_VMM=1
export TILELANG_USE_DISTRIBUTED=1
