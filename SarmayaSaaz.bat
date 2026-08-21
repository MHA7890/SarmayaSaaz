@echo off
title SarmayaSaaz Launcher
if exist "SarmayaSaaz_Launcher.exe" (
    start "" "SarmayaSaaz_Launcher.exe"
) else if exist "SarmayaSaaz_Launcher.pyw" (
    start "" pythonw "SarmayaSaaz_Launcher.pyw"
) else (
    echo Error: Could not find SarmayaSaaz_Launcher executable or python script.
    pause
)
