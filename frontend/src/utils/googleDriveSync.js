/**
 * Google Drive API v3 Sync Utility for Voication Studio
 * Uses Google Identity Services (GIS) with narrow 'drive.file' scope.
 */

const DRIVE_FOLDER_NAME = "Voication_Studio_Backups";
const DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile";

let gisScriptLoaded = false;
let tokenClient = null;

/**
 * Dynamically load Google Identity Services (GIS) client library
 */
export const loadGisScript = () => {
  return new Promise((resolve, reject) => {
    if (gisScriptLoaded || window.google?.accounts?.oauth2) {
      gisScriptLoaded = true;
      return resolve(true);
    }
    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.defer = true;
    script.onload = () => {
      gisScriptLoaded = true;
      resolve(true);
    };
    script.onerror = (err) => reject(err);
    document.body.appendChild(script);
  });
};

/**
 * Initialize token client and request access token via Google Popup
 */
export const requestDriveAccessToken = async (clientId, onTokenReceived, onError) => {
  try {
    await loadGisScript();

    if (!clientId) {
      clientId = "1041584982401-voicationstudiomock.apps.googleusercontent.com"; // Default fallback
    }

    if (!window.google?.accounts?.oauth2) {
      throw new Error("Google Identity Services failed to initialize.");
    }

    tokenClient = window.google.accounts.oauth2.initTokenClient({
      client_id: clientId,
      scope: DRIVE_SCOPE,
      callback: async (response) => {
        if (response.error) {
          if (onError) onError(response.error);
          return;
        }
        if (response.access_token) {
          // Fetch user info for display
          let userEmail = "";
          try {
            const userRes = await fetch("https://www.googleapis.com/oauth2/v3/userinfo", {
              headers: { Authorization: `Bearer ${response.access_token}` }
            });
            if (userRes.ok) {
              const userData = await userRes.json();
              userEmail = userData.email || userData.name || "";
            }
          } catch (e) {
            console.warn("Could not fetch user email:", e);
          }

          if (onTokenReceived) {
            onTokenReceived({
              accessToken: response.access_token,
              expiresIn: response.expires_in,
              userEmail
            });
          }
        }
      }
    });

    tokenClient.requestAccessToken({ prompt: "consent" });
  } catch (err) {
    if (onError) onError(err.message || err);
  }
};

/**
 * Find or create the 'Voication_Studio_Backups' folder in user's Drive
 */
export const ensureBackupFolder = async (accessToken) => {
  const query = encodeURIComponent(`name = '${DRIVE_FOLDER_NAME}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false`);
  const searchUrl = `https://www.googleapis.com/drive/v3/files?q=${query}&fields=files(id,name)`;

  const res = await fetch(searchUrl, {
    headers: { Authorization: `Bearer ${accessToken}` }
  });

  if (!res.ok) {
    throw new Error(`Failed to query Drive folder: ${res.statusText}`);
  }

  const data = await res.json();
  if (data.files && data.files.length > 0) {
    return data.files[0].id;
  }

  // Create folder if missing
  const createRes = await fetch("https://www.googleapis.com/drive/v3/files", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      name: DRIVE_FOLDER_NAME,
      mimeType: "application/vnd.google-apps.folder"
    })
  });

  if (!createRes.ok) {
    throw new Error(`Failed to create Drive folder: ${createRes.statusText}`);
  }

  const folderData = await createRes.json();
  return folderData.id;
};

/**
 * Upload or update a single project JSON in Google Drive
 */
export const uploadProjectToDrive = async (project, accessToken) => {
  const folderId = await ensureBackupFolder(accessToken);
  const fileName = `Voication_${project.name.replace(/[^a-zA-Z0-9_-]/g, "_")}_${project.id}.json`;

  // Search if file already exists in folder
  const query = encodeURIComponent(`name = '${fileName}' and '${folderId}' in parents and trashed = false`);
  const searchUrl = `https://www.googleapis.com/drive/v3/files?q=${query}&fields=files(id,name)`;

  const searchRes = await fetch(searchUrl, {
    headers: { Authorization: `Bearer ${accessToken}` }
  });

  let existingFileId = null;
  if (searchRes.ok) {
    const searchData = await searchRes.json();
    if (searchData.files && searchData.files.length > 0) {
      existingFileId = searchData.files[0].id;
    }
  }

  const backupPayload = {
    ...project,
    cloudBackedUpAt: new Date().toISOString(),
    version: "1.0"
  };

  const fileContent = JSON.stringify(backupPayload, null, 2);
  const metadata = {
    name: fileName,
    mimeType: "application/json",
    parents: existingFileId ? undefined : [folderId]
  };

  const boundary = "-------314159265358979323846";
  const delimiter = `\r\n--${boundary}\r\n`;
  const closeDelimiter = `\r\n--${boundary}--`;

  const multipartRequestBody =
    delimiter +
    "Content-Type: application/json\r\n\r\n" +
    JSON.stringify(metadata) +
    delimiter +
    "Content-Type: application/json\r\n\r\n" +
    fileContent +
    closeDelimiter;

  const uploadUrl = existingFileId
    ? `https://www.googleapis.com/upload/drive/v3/files/${existingFileId}?uploadType=multipart`
    : `https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart`;

  const uploadRes = await fetch(uploadUrl, {
    method: existingFileId ? "PATCH" : "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": `multipart/related; boundary=${boundary}`
    },
    body: multipartRequestBody
  });

  if (!uploadRes.ok) {
    throw new Error(`Upload failed: ${uploadRes.statusText}`);
  }

  return await uploadRes.json();
};

/**
 * List all backup files in Google Drive folder
 */
export const fetchDriveBackups = async (accessToken) => {
  const folderId = await ensureBackupFolder(accessToken);
  const query = encodeURIComponent(`'${folderId}' in parents and trashed = false`);
  const listUrl = `https://www.googleapis.com/drive/v3/files?q=${query}&fields=files(id,name,mimeType,createdTime,modifiedTime,size)&orderBy=modifiedTime desc`;

  const res = await fetch(listUrl, {
    headers: { Authorization: `Bearer ${accessToken}` }
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch Drive backups: ${res.statusText}`);
  }

  const data = await res.json();
  return data.files || [];
};

/**
 * Download a project JSON backup file content from Google Drive
 */
export const downloadDriveProject = async (fileId, accessToken) => {
  const downloadUrl = `https://www.googleapis.com/drive/v3/files/${fileId}?alt=media`;

  const res = await fetch(downloadUrl, {
    headers: { Authorization: `Bearer ${accessToken}` }
  });

  if (!res.ok) {
    throw new Error(`Failed to download file from Drive: ${res.statusText}`);
  }

  const projectData = await res.json();
  return projectData;
};
