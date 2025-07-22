from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_Meteorology, TIDE_LEVEL_15MIN_MULTI, Dataset_Pred
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'TIDE': TIDE_LEVEL_15MIN_MULTI,
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'Meteorology' : Dataset_Meteorology
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    # ★★★ 핵심 수정 사항 1 ★★★
    # val, test, test_full 에서는 shuffle을 False로 설정
    shuffle_flag = False if flag in ['test', 'TEST', 'val', 'test_full'] else True
    # train일 때만 마지막 불완전한 배치를 버리고, 나머지는 모두 사용
    drop_last = True if flag == 'train' else False
    # --------------------------

    batch_size = args.batch_size
    freq = args.freq
        
    # (if/elif/else 로직은 사용자 환경에 맞게 유지하되, 아래 구조를 따릅니다)
    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
        
    return data_set, data_loader