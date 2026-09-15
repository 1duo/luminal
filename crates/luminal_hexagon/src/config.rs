use std::path::PathBuf;

/// Host-side settings for a Hexagon SDK/FastRPC session.
#[derive(Debug, Clone)]
pub struct HexagonConfig {
    /// Optional path to `libcdsprpc.dll`. If omitted, the platform loader is
    /// used and `LUMINAL_HEXAGON_RPC_DLL` is consulted first.
    pub rpc_library: Option<PathBuf>,
    /// FastRPC URI for the installed v73 skel.
    pub skel_uri: String,
    /// FastRPC domain. Snapdragon X Elite's compute DSP is domain 3 (CDSP).
    pub domain: i32,
    /// Allow an unsigned DSP module when the device permits it. Production
    /// deployments should install a signed/catalogued module instead.
    pub allow_unsigned_module: bool,
}

impl Default for HexagonConfig {
    fn default() -> Self {
        Self {
            rpc_library: std::env::var_os("LUMINAL_HEXAGON_RPC_DLL").map(PathBuf::from),
            skel_uri: std::env::var("LUMINAL_HEXAGON_SKEL_URI").unwrap_or_else(|_| {
                "file:///libluminal_hexagon-v73.so?luminal_hexagon_skel_handle_invoke&_modver=1.0&_dom=cdsp"
                    .to_string()
            }),
            domain: 3,
            allow_unsigned_module: true,
        }
    }
}

impl HexagonConfig {
    pub fn with_rpc_library(mut self, path: impl Into<PathBuf>) -> Self {
        self.rpc_library = Some(path.into());
        self
    }

    pub fn with_skel_uri(mut self, uri: impl Into<String>) -> Self {
        self.skel_uri = uri.into();
        self
    }
}
