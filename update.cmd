@chcp 65001 >nul
@echo off
title 更新腳本 (Update Script)
echo 正在從 GitHub 拉取最新版本的程式碼...
echo.

rem 執行 git pull，它會使用您本地分支跟蹤的遠端分支
git pull

echo.
echo ==================================================================
echo 更新完成！
echo.
echo - 如果看到 'Already up to date.' 表示您已經是最新版本。
echo - 如果看到檔案衝突 ^(conflict^) 的錯誤，請先解決衝突或聯繫開發者。
echo ==================================================================
echo.


echo 請按任意鍵結束...
pause >nul