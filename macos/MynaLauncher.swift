import AppKit
import Darwin
import Foundation
import WebKit

private let appName = "Myna"
private let host = "127.0.0.1"
private let port = 4180
private let healthURL = URL(string: "http://\(host):\(port)/api/health")!
private let appURL = URL(string: "http://\(host):\(port)")!

private func projectRootURL() -> URL {
    Bundle.main.bundleURL.deletingLastPathComponent()
}

private func showAlert(title: String, message: String) {
    DispatchQueue.main.async {
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.addButton(withTitle: "好")
        alert.runModal()
    }
}

private func checkHealth(timeout: TimeInterval = 1.2, completion: @escaping (Bool) -> Void) {
    var request = URLRequest(url: healthURL)
    request.timeoutInterval = timeout
    URLSession.shared.dataTask(with: request) { _, response, _ in
        let ok = (response as? HTTPURLResponse)?.statusCode == 200
        completion(ok)
    }.resume()
}

private final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate, WKUIDelegate, WKDownloadDelegate {
    private let rootURL = projectRootURL()
    private var serverProcess: Process?
    private var logHandle: FileHandle?
    private var readinessTimer: Timer?
    private var readinessAttempts = 0
    private var openedMainWindow = false
    private var startedByThisApp = false
    private var adoptedServerPID: Int32?
    private var mainWindow: NSWindow?
    private var webView: WKWebView?

    private var managesServer: Bool {
        startedByThisApp || adoptedServerPID != nil
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        installMenu()
        startOrOpen()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    // 点 Dock 图标时触发：服务活着直接打开桌面窗口，挂了则重启
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        startOrOpen()
        return false
    }

    // Dock 右键菜单
    func applicationDockMenu(_ sender: NSApplication) -> NSMenu? {
        let menu = NSMenu()
        let openItem = NSMenuItem(title: "打开 Myna", action: #selector(startOrOpenFromMenu), keyEquivalent: "")
        openItem.target = self
        menu.addItem(openItem)
        let browserItem = NSMenuItem(title: "在浏览器中打开", action: #selector(openInBrowser), keyEquivalent: "")
        browserItem.target = self
        menu.addItem(browserItem)
        return menu
    }

    // 退出前检查是否有训练任务在运行
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard managesServer else { return .terminateNow }

        var request = URLRequest(url: URL(string: "http://\(host):\(port)/api/queue")!)
        request.timeoutInterval = 1.5
        let semaphore = DispatchSemaphore(value: 0)
        var hasRunning = false

        URLSession.shared.dataTask(with: request) { data, _, _ in
            if let data,
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let runningId = json["running_id"],
               !(runningId is NSNull) {
                hasRunning = true
            }
            semaphore.signal()
        }.resume()
        semaphore.wait()

        guard hasRunning else { return .terminateNow }

        DispatchQueue.main.async {
            NSApp.activate(ignoringOtherApps: true)
            let alert = NSAlert()
            alert.messageText = "训练任务正在进行"
            alert.informativeText = "当前有一个模型训练任务正在运行。现在退出将中断训练，已训练的进度会丢失。"
            alert.alertStyle = .warning
            alert.addButton(withTitle: "继续退出")
            alert.addButton(withTitle: "取消")
            let response = alert.runModal()
            if response == .alertFirstButtonReturn {
                NSApp.reply(toApplicationShouldTerminate: true)
            } else {
                NSApp.reply(toApplicationShouldTerminate: false)
            }
        }
        return .terminateLater
    }

    func applicationWillTerminate(_ notification: Notification) {
        readinessTimer?.invalidate()
        terminateManagedServer()
        try? logHandle?.close()
    }

    private func installMenu() {
        let mainMenu = NSMenu()
        let appMenuItem = NSMenuItem()
        mainMenu.addItem(appMenuItem)

        let appMenu = NSMenu(title: appName)
        let openItem = NSMenuItem(title: "打开 Myna", action: #selector(startOrOpenFromMenu), keyEquivalent: "o")
        openItem.target = self
        appMenu.addItem(openItem)
        let browserItem = NSMenuItem(title: "在浏览器中打开", action: #selector(openInBrowser), keyEquivalent: "b")
        browserItem.target = self
        appMenu.addItem(browserItem)
        appMenu.addItem(.separator())
        appMenu.addItem(
            NSMenuItem(
                title: "退出 Myna",
                action: #selector(NSApplication.terminate(_:)),
                keyEquivalent: "q"
            )
        )
        appMenuItem.submenu = appMenu
        NSApp.mainMenu = mainMenu
    }

    private func startOrOpen() {
        checkHealth { [weak self] healthy in
            guard let self else { return }
            DispatchQueue.main.async {
                if healthy {
                    self.adoptExistingServerIfNeeded()
                    self.openMyna()
                    return
                }
                self.startServer()
            }
        }
    }

    private func startServer() {
        let pythonURL = rootURL.appendingPathComponent(".venv/bin/python")
        let startScriptURL = rootURL.appendingPathComponent("scripts/start.command")

        guard FileManager.default.isExecutableFile(atPath: pythonURL.path) else {
            let installHint = FileManager.default.fileExists(atPath: startScriptURL.path)
                ? "请先在项目目录运行 scripts/install.command 完成安装。旧入口「启动Myna.command」也仍可作为备用。"
                : "请确认 Myna.app 仍放在项目根目录，和 scripts、local_trainer、web 在同一层。"
            showAlert(title: "Myna 启动失败", message: "没有找到运行环境 .venv。\n\n\(installHint)")
            NSApp.terminate(nil)
            return
        }

        let runtimeURL = rootURL.appendingPathComponent("runtime", isDirectory: true)
        let logURL = runtimeURL.appendingPathComponent("myna-app.log")
        do {
            try FileManager.default.createDirectory(at: runtimeURL, withIntermediateDirectories: true)
            if !FileManager.default.fileExists(atPath: logURL.path) {
                FileManager.default.createFile(atPath: logURL.path, contents: nil)
            }
            logHandle = try FileHandle(forWritingTo: logURL)
            try logHandle?.seekToEnd()
        } catch {
            showAlert(title: "Myna 启动失败", message: "无法写入日志文件：\(error.localizedDescription)")
            NSApp.terminate(nil)
            return
        }

        let process = Process()
        process.executableURL = pythonURL
        process.currentDirectoryURL = rootURL
        process.arguments = [
            "-m",
            "uvicorn",
            "local_trainer.main:app",
            "--host",
            host,
            "--port",
            String(port),
        ]
        var environment = ProcessInfo.processInfo.environment
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        process.environment = environment
        process.standardOutput = logHandle
        process.standardError = logHandle
        process.terminationHandler = { [weak self] finishedProcess in
            DispatchQueue.main.async {
                guard let self else { return }
                self.readinessTimer?.invalidate()
                if !self.openedMainWindow {
                    showAlert(
                        title: "Myna 启动失败",
                        message: "本地服务已退出（状态码 \(finishedProcess.terminationStatus)）。请查看 runtime/myna-app.log，或运行 scripts/doctor.command 检查环境。"
                    )
                }
            }
        }

        do {
            try process.run()
            serverProcess = process
            startedByThisApp = true
            waitUntilReady()
        } catch {
            showAlert(title: "Myna 启动失败", message: "无法启动本地服务：\(error.localizedDescription)")
            NSApp.terminate(nil)
        }
    }

    private func waitUntilReady() {
        readinessAttempts = 0
        readinessTimer?.invalidate()
        readinessTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] timer in
            guard let self else { return }
            self.readinessAttempts += 1
            checkHealth { healthy in
                DispatchQueue.main.async {
                    if healthy {
                        timer.invalidate()
                        self.openMyna()
                    } else if self.readinessAttempts >= 120 {
                        timer.invalidate()
                        showAlert(
                            title: "Myna 启动超时",
                            message: "本地服务 60 秒内没有就绪。请查看 runtime/myna-app.log，或运行 scripts/doctor.command 检查环境。"
                        )
                    }
                }
            }
        }
    }

    @objc private func startOrOpenFromMenu() {
        startOrOpen()
    }

    @objc private func openMyna() {
        openedMainWindow = true
        showMainWindow()
    }

    @objc private func openInBrowser() {
        NSWorkspace.shared.open(appURL)
    }

    private func adoptExistingServerIfNeeded() {
        guard !startedByThisApp, serverProcess == nil, adoptedServerPID == nil else { return }
        adoptedServerPID = listeningServerPID()
    }

    private func listeningServerPID() -> Int32? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
        process.arguments = ["-tiTCP:\(port)", "-sTCP:LISTEN"]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()

        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return nil
        }

        guard process.terminationStatus == 0 else { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""
        return output
            .split(whereSeparator: \.isNewline)
            .compactMap { Int32($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
            .first
    }

    private func terminateManagedServer() {
        if startedByThisApp, let process = serverProcess, process.isRunning {
            process.terminate()
            waitForServerToExit(pid: process.processIdentifier)
            if process.isRunning {
                kill(process.processIdentifier, SIGKILL)
            }
        }

        if let pid = adoptedServerPID, isProcessAlive(pid) {
            kill(pid, SIGTERM)
            waitForServerToExit(pid: pid)
            if isProcessAlive(pid) {
                kill(pid, SIGKILL)
            }
        }
    }

    private func waitForServerToExit(pid: Int32) {
        for _ in 0..<20 {
            if !isProcessAlive(pid) || listeningServerPID() == nil {
                return
            }
            Thread.sleep(forTimeInterval: 0.1)
        }
    }

    private func isProcessAlive(_ pid: Int32) -> Bool {
        kill(pid, 0) == 0
    }

    private func showMainWindow() {
        NSApp.activate(ignoringOtherApps: true)

        if mainWindow == nil {
            createMainWindow()
        }

        guard let window = mainWindow, let webView else { return }
        if webView.url == nil {
            webView.load(URLRequest(url: appURL))
        }
        window.makeKeyAndOrderFront(nil)
    }

    private func createMainWindow() {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 860),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = appName
        window.center()
        window.minSize = NSSize(width: 1024, height: 680)
        window.isReleasedWhenClosed = false
        window.delegate = self

        let webView = WKWebView(frame: window.contentView?.bounds ?? .zero, configuration: configuration)
        webView.autoresizingMask = [.width, .height]
        webView.navigationDelegate = self
        webView.uiDelegate = self
        window.contentView = webView

        mainWindow = window
        self.webView = webView
    }

    private func isLocalAppURL(_ url: URL) -> Bool {
        url.scheme == "http" && url.host == host && url.port == port
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        sender.orderOut(nil)
        return false
    }

    // target="_blank" 链接：本地 URL 在当前 webView 打开，外部 URL 用浏览器
    func webView(
        _ webView: WKWebView,
        createWebViewWith configuration: WKWebViewConfiguration,
        for navigationAction: WKNavigationAction,
        windowFeatures: WKWindowFeatures
    ) -> WKWebView? {
        guard let url = navigationAction.request.url else { return nil }
        if isLocalAppURL(url) {
            webView.load(URLRequest(url: url))
        } else {
            NSWorkspace.shared.open(url)
        }
        return nil
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
    ) {
        guard let url = navigationAction.request.url else {
            decisionHandler(.allow)
            return
        }
        if navigationAction.targetFrame?.isMainFrame == false || isLocalAppURL(url) {
            decisionHandler(.allow)
            return
        }
        if url.scheme == "http" || url.scheme == "https" {
            NSWorkspace.shared.open(url)
            decisionHandler(.cancel)
            return
        }
        decisionHandler(.allow)
    }

    // 检测下载响应（Content-Disposition: attachment 或非网页 MIME）
    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationResponse: WKNavigationResponse,
        decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void
    ) {
        if let response = navigationResponse.response as? HTTPURLResponse,
           let disposition = response.value(forHTTPHeaderField: "Content-Disposition"),
           disposition.lowercased().contains("attachment") {
            decisionHandler(.download)
            return
        }
        if !navigationResponse.canShowMIMEType {
            decisionHandler(.download)
            return
        }
        decisionHandler(.allow)
    }

    // 导航转下载时触发
    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload) {
        download.delegate = self
    }

    func webView(_ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload) {
        download.delegate = self
    }

    // WKDownloadDelegate：弹出保存面板让用户选择保存位置
    func download(_ download: WKDownload, decideDestinationUsing response: URLResponse, suggestedFilename: String, completionHandler: @escaping (URL?) -> Void) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = suggestedFilename
        panel.canCreateDirectories = true
        panel.begin { result in
            completionHandler(result == .OK ? panel.url : nil)
        }
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        let pageTitle = webView.title ?? ""
        mainWindow?.title = pageTitle.isEmpty ? appName : "\(appName) - \(pageTitle)"
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptAlertPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping () -> Void
    ) {
        let alert = NSAlert()
        alert.messageText = appName
        alert.informativeText = message
        alert.alertStyle = .informational
        alert.addButton(withTitle: "好")
        alert.runModal()
        completionHandler()
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptConfirmPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping (Bool) -> Void
    ) {
        let alert = NSAlert()
        alert.messageText = appName
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.addButton(withTitle: "确认")
        alert.addButton(withTitle: "取消")
        completionHandler(alert.runModal() == .alertFirstButtonReturn)
    }

    func webView(
        _ webView: WKWebView,
        runOpenPanelWith parameters: WKOpenPanelParameters,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping ([URL]?) -> Void
    ) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.canChooseDirectories = parameters.allowsDirectories
        panel.canChooseFiles = true
        panel.begin { result in
            completionHandler(result == .OK ? panel.urls : nil)
        }
    }
}

if ProcessInfo.processInfo.environment["MYNA_APP_DRY_RUN"] == "1" {
    let root = projectRootURL()
    print(root.path)
    print(root.appendingPathComponent(".venv/bin/python").path)
    print(root.appendingPathComponent("runtime/myna-app.log").path)
    exit(0)
}

let app = NSApplication.shared
private let delegate = AppDelegate()
app.delegate = delegate
app.run()
