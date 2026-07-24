param(
    [string]$OutputPath = (Join-Path $PSScriptRoot 'gate0-source.png')
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$width = 900
$height = 1200
$bitmap = [System.Drawing.Bitmap]::new(
    $width,
    $height,
    [System.Drawing.Imaging.PixelFormat]::Format32bppArgb
)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit

$white = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::White)
$paper = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(244, 244, 242))
$ink = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(20, 20, 20))
$panelPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(24, 24, 24), 8)
$bubblePen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(18, 18, 18), 5)
$detailPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(90, 90, 90), 4)
$dialogueFont = [System.Drawing.Font]::new(
    'Arial',
    31,
    [System.Drawing.FontStyle]::Bold,
    [System.Drawing.GraphicsUnit]::Pixel
)
$captionFont = [System.Drawing.Font]::new(
    'Arial',
    25,
    [System.Drawing.FontStyle]::Regular,
    [System.Drawing.GraphicsUnit]::Pixel
)
$format = [System.Drawing.StringFormat]::new()
$format.Alignment = [System.Drawing.StringAlignment]::Center
$format.LineAlignment = [System.Drawing.StringAlignment]::Center

try {
    $graphics.FillRectangle($white, 0, 0, $width, $height)

    $top = [System.Drawing.Rectangle]::new(35, 35, 830, 520)
    $bottomLeft = [System.Drawing.Rectangle]::new(35, 590, 400, 575)
    $bottomRight = [System.Drawing.Rectangle]::new(465, 590, 400, 575)
    foreach ($panel in @($top, $bottomLeft, $bottomRight)) {
        $graphics.FillRectangle($paper, $panel)
        $graphics.DrawRectangle($panelPen, $panel)
    }

    # Simple synthetic figures and motion lines make the page less like a
    # plain OCR card while remaining wholly generated for this repository.
    $graphics.DrawEllipse($detailPen, 90, 235, 145, 145)
    $graphics.DrawLine($detailPen, 160, 380, 140, 510)
    $graphics.DrawLine($detailPen, 160, 420, 95, 505)
    $graphics.DrawLine($detailPen, 160, 420, 230, 505)
    $graphics.DrawEllipse($detailPen, 520, 250, 145, 145)
    $graphics.DrawLine($detailPen, 590, 395, 610, 520)
    for ($index = 0; $index -lt 9; $index++) {
        $offset = $index * 27
        $graphics.DrawLine($detailPen, 505 + $offset, 470, 460 + $offset, 540)
    }

    $bubbleOne = [System.Drawing.Rectangle]::new(255, 90, 510, 190)
    $bubbleTwo = [System.Drawing.Rectangle]::new(75, 640, 320, 185)
    $bubbleThree = [System.Drawing.Rectangle]::new(505, 870, 320, 190)
    foreach ($bubble in @($bubbleOne, $bubbleTwo, $bubbleThree)) {
        $graphics.FillEllipse($white, $bubble)
        $graphics.DrawEllipse($bubblePen, $bubble)
    }
    $graphics.DrawLine($bubblePen, 350, 258, 315, 330)
    $graphics.DrawLine($bubblePen, 350, 258, 395, 300)
    $graphics.DrawLine($bubblePen, 300, 800, 345, 870)
    $graphics.DrawLine($bubblePen, 300, 800, 260, 840)
    $graphics.DrawLine($bubblePen, 600, 1035, 555, 1110)
    $graphics.DrawLine($bubblePen, 600, 1035, 650, 1085)

    $graphics.DrawString(
        "WE HAVE TO`nLEAVE NOW!",
        $dialogueFont,
        $ink,
        [System.Drawing.RectangleF]::new(285, 118, 450, 125),
        $format
    )
    $graphics.DrawString(
        "ARE YOU`nREADY?",
        $dialogueFont,
        $ink,
        [System.Drawing.RectangleF]::new(100, 675, 270, 115),
        $format
    )
    $graphics.DrawString(
        "YES.`nLET'S GO!",
        $dialogueFont,
        $ink,
        [System.Drawing.RectangleF]::new(535, 905, 260, 120),
        $format
    )
    $graphics.DrawString(
        'A synthetic Gate 0 fixture',
        $captionFont,
        $ink,
        [System.Drawing.RectangleF]::new(500, 1090, 320, 42),
        $format
    )

    $parent = Split-Path -Parent $OutputPath
    if ($parent) {
        [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    }
    $bitmap.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
    Write-Output $OutputPath
}
finally {
    $format.Dispose()
    $captionFont.Dispose()
    $dialogueFont.Dispose()
    $detailPen.Dispose()
    $bubblePen.Dispose()
    $panelPen.Dispose()
    $ink.Dispose()
    $paper.Dispose()
    $white.Dispose()
    $graphics.Dispose()
    $bitmap.Dispose()
}
