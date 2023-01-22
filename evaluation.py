import csv
import numpy as np
import os

def file_comapre(file_gauge, file): #output the lists of MAE and RMSE
    reader_g = csv.reader(open(file_gauge))
    reader = csv.reader(open(file))
    gauge_dict = dict()
    maelist = list()
    rmselist = list()
    for row in reader_g:
        gauge_dict[(row[0],row[1])] = float(row[2])
    for row_ in reader:
        if (row_[0],row_[1]) in gauge_dict:
            if float(row_[2]) == 9998.95: # there are some errors in grid data that needs to skip
                continue
            error = gauge_dict[(row_[0],row_[1])] - float(row_[2])
            maelist.append(np.abs(error))
            rmselist.append(error**2)
    return maelist, rmselist

def hourly_compare(partition, file):
    cmpas = 'CMPAS/' + file
    tmf = 'TMF/' + partition + '/'+ file
    gauge = 'gauge-partition/' + partition + '/Test/' + file
    maelist, rmselist = file_comapre(gauge, cmpas)
    mae_cmpas = round(sum(maelist) / len(maelist),4)
    rmse_cmpas = round(np.sqrt(sum(rmselist) / len(rmselist)),4)
    maelist, rmselist = file_comapre(gauge, tmf)
    mae_tmf = round(sum(maelist) / len(maelist),4)
    rmse_tmf = round(np.sqrt(sum(rmselist) / len(rmselist)),4)
    return mae_cmpas, mae_tmf, rmse_cmpas, rmse_tmf

def longterm_compare(partition, filelist):
    mae_cmpas_list = list()
    rmse_cmpas_list = list()
    mae_tmf_list = list()
    rmse_tmf_list = list()
    for file in filelist:
        cmpas = 'CMPAS/' + file
        tmf = 'TMF/' + partition + '/'+ file
        gauge = 'gauge-partition/' + partition + '/Test/' + file
        maelist, rmselist = file_comapre(gauge, cmpas)
        mae_cmpas_list = mae_cmpas_list + maelist
        rmse_cmpas_list = rmse_cmpas_list + rmselist
        maelist, rmselist = file_comapre(gauge, tmf)
        mae_tmf_list = mae_tmf_list + maelist
        rmse_tmf_list = rmse_tmf_list + rmselist
    mae_cmpas = round(sum(mae_cmpas_list) / len(mae_cmpas_list),4)
    rmse_cmpas = round(np.sqrt(sum(rmse_cmpas_list) / len(rmse_cmpas_list)),4)
    mae_tmf = round(sum(mae_tmf_list) / len(mae_tmf_list),4)
    rmse_tmf = round(np.sqrt(sum(rmse_tmf_list) / len(rmse_tmf_list)),4)
    return mae_cmpas, mae_tmf, rmse_cmpas, rmse_tmf

if __name__ == '__main__':
    partition = '10%' # 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%
    print('training partition: '+partition)
    daydict = dict()
    monthdict = dict()
    totallist = list()
    print('hourly')
    for file in os.listdir('CMPAS/'):
        totallist.append(file)
        mae_cmpas, mae_tmf, rmse_cmpas, rmse_tmf = hourly_compare(partition, file)
        writer_hou = csv.writer(open('evaluation_results/hourly/train_'+partition+'.csv','a',newline=''))
        writer_hou.writerow([file[0:10], mae_cmpas, mae_tmf, rmse_cmpas, rmse_tmf])
        day = file[0:8]
        month = file[0:6]
        if day not in daydict:
            daydict[day] = list()
        if month not in monthdict:
            monthdict[month] = list()
        daydict[day].append(file)
        monthdict[month].append(file)

    print('----------')
    print('daily')
    for day in daydict:
        mae_cmpas, mae_tmf, rmse_cmpas, rmse_tmf = longterm_compare(partition, daydict[day])
        writer_day =csv.writer(open('evaluation_results/daily/train_'+partition+'.csv','a',newline=''))
        writer_day.writerow([day, mae_cmpas, mae_tmf, rmse_cmpas, rmse_tmf])

    print('----------')
    print('monthly')
    for month in monthdict:
        mae_cmpas, mae_tmf, rmse_cmpas, rmse_tmf = longterm_compare(partition, monthdict[month])
        print([mae_cmpas, mae_tmf, rmse_cmpas, rmse_tmf])
        writer_mon = csv.writer(open('evaluation_results/monthly/train_' + partition + '.csv', 'a', newline=''))
        writer_mon.writerow([month, mae_cmpas, mae_tmf, rmse_cmpas, rmse_tmf])

    print('----------')
    print('total')
    mae_cmpas, mae_tmf, rmse_cmpas, rmse_tmf = longterm_compare(partition, totallist)
    print([mae_cmpas, mae_tmf, rmse_cmpas, rmse_tmf])
    writer_tot = csv.writer(open('evaluation_results/total/train_' + partition + '.csv', 'a', newline=''))
    writer_tot.writerow([mae_cmpas, mae_tmf, rmse_cmpas, rmse_tmf])