# W0-00 硬體探測。RAM 通道數是最高 ROI 的 gate（單→雙通道頻寬 +30-50%，直接決定 CPU 推論速度）。
# 用法：powershell -ExecutionPolicy Bypass -File 00_hardware.ps1

Write-Output "===== CPU ====="
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed | Format-List

Write-Output "===== RAM 總量 ====="
"{0} GB" -f [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)

Write-Output "===== RAM 插槽 / 通道（關鍵）====="
$dimms = @(Get-CimInstance Win32_PhysicalMemory)
$dimms | Select-Object @{n='Slot';e={$_.DeviceLocator}}, @{n='SizeGB';e={[math]::Round($_.Capacity/1GB,0)}}, @{n='SpeedMTs';e={$_.Speed}}, Manufacturer, PartNumber | Format-Table -AutoSize
Write-Output ("已插記憶體條數：{0}" -f $dimms.Count)
if ($dimms.Count -ge 2) {
  Write-Output "→ 可能為雙通道(dual-channel)。建議再用 CPU-Z 的 Memory 分頁確認 Channel #（# of Channels = 2 才是真雙通道）。"
} else {
  Write-Output "→ 僅 1 條 = 單通道(single-channel)。這是 CPU 推論的頻寬瓶頸；若插槽可加、補一條同規 RAM 是最高 ROI。"
}

Write-Output "===== 記憶體插槽總數（看還能不能加）====="
Get-CimInstance Win32_PhysicalMemoryArray | Select-Object @{n='TotalSlots';e={$_.MemoryDevices}} | Format-List

Write-Output "===== GPU ====="
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion | Format-List

Write-Output "===== 磁碟可用 ====="
Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | Select-Object DeviceID, @{n='FreeGB';e={[math]::Round($_.FreeSpace/1GB,1)}}, @{n='TotalGB';e={[math]::Round($_.Size/1GB,1)}} | Format-Table -AutoSize
